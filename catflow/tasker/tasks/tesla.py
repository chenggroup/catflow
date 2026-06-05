import json
import os
import sys
import shutil
import warnings
from glob import glob
from pathlib import Path

from yaml import load, SafeLoader
import numpy as np

from ase.io import read, write

from catflow.utils import logger
from catflow.utils.file import count_lines
from catflow.tasker.resources.config import MachineConfig, SbatchResources
from catflow.tasker.resources.script_gen import TeslaBatchScript
from catflow.tasker.resources.workflow_executor import WorkflowExecutor
from catflow.tasker.resources.template_engine import render_template, find_template


def _require_dpgen():
    """Lazy-load dpgen module with a clear error if not installed.

    Returns the dpgen module if available, or raises ImportError with
    instructions to use CATFLOW_USE_TEMPLATE=1 / CATFLOW_USE_OMB=1.
    """
    try:
        import dpgen as _dpgen
        return _dpgen
    except ImportError:
        raise ImportError(
            "dpgen is not installed. Set CATFLOW_USE_TEMPLATE=1 and "
            "CATFLOW_USE_OMB=1 to use the oh-my-batch + template workflow "
            "without dpgen. Alternatively, install dpgen: pip install dpgen"
        )

class TeslaWorkStep(object):
    def __init__(self, params, step_code, machine):
        self.params = params
        self.step_code = step_code
        self.machine = machine

    @property
    def sub_step_dict(self):
        return {
            0: self.make,
            1: self.run,
            2: self.post
        }

    def make(self):
        pass

    def run(self):
        pass

    def post(self):
        pass

    def _use_template(self) -> bool:
        """Check if template-based input generation should be used."""
        return os.environ.get("CATFLOW_USE_TEMPLATE", "0") == "1"

    def _build_workflow_executor(self, work_dir: str) -> WorkflowExecutor:
        """Build a WorkflowExecutor from the dpgen machine config.

        Attempts to extract scheduler and resource info from the
        standard dpgen machine config format.
        """
        try:
            mdata = self.machine
            # Try to get first machine config for submission
            if isinstance(mdata, dict) and len(mdata) > 0:
                first_key = list(mdata.keys())[0]
                mcfg = mdata[first_key] if isinstance(mdata, dict) else {}
                return WorkflowExecutor.local(work_base=work_dir)
        except Exception:
            pass
        return WorkflowExecutor.local(work_base=work_dir)


# TODO: refine each step of whole workflow.
class DPTrain(TeslaWorkStep):

    def make(self):
        if self._use_template():
            return self._make_template()
        _require_dpgen().generator.run.make_train(
            self.step_code, self.params, self.machine)

    def _make_template(self):
        """Generate DeePMD training input files from templates."""
        import dpdata
        from ase.io import read as ase_read

        iter_idx = self.step_code
        iter_dir = Path(f"iter.{str(iter_idx).zfill(6)}")
        train_dir = iter_dir / "00.train"
        train_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[TMPL] Generating training tasks in {train_dir}")

        # --- data setup ---
        init_data_path = Path(self.params.get("init_data_prefix", "."))
        init_data_name = self.params.get("init_data", "")
        train_dir_symlink = train_dir / "data.init"
        if not train_dir_symlink.exists():
            target = (init_data_path / init_data_name).resolve()
            train_dir_symlink.symlink_to(target, target_is_directory=True)

        # --- type_map & sel ---
        type_map = self.params.get("type_map", [])
        type_map_str = ", ".join(f'"{t}"' for t in type_map)

        # Calculate sel from type_map based on rcut
        try:
            from dpdata.system import get_frame
            sys_data = dpdata.LabeledSystem(str(train_dir_symlink), fmt="deepmd/npy")
            type_count = dict(zip(type_map, [0]*len(type_map)))
            for atom_type in sys_data['atom_types']:
                type_count[type_map[atom_type]] = type_count.get(type_map[atom_type], 0) + 1
            sel = [max(1, int(c * 1.2)) for c in type_count.values()]
            sel_str = ", ".join(str(s) for s in sel)
        except Exception:
            sel_str = "46"

        # --- dataset paths ---
        dataset_paths = [f'"{train_dir}/data.init"']
        for i in range(iter_idx):
            prev_data = Path(f"iter.{str(i).zfill(6)}") / "02.fp" / "data"
            if prev_data.exists():
                dataset_paths.append(f'"{prev_data}"')
        dataset_str = ", ".join(dataset_paths)

        # --- model count & seeds ---
        numb_models = self.params.get("numb_models", 4)
        seeds = list(np.random.randint(0, 1000000, size=numb_models))

        # --- training steps ---
        default_param = self.params.get("default_training_param", {})
        n_steps = default_param.get("training", {}).get("numb_steps", 400000)
        decay_steps = default_param.get("learning_rate", {}).get("decay_steps", 5000)

        # --- generate per-model input.json ---
        tpl_path = find_template("deepmd/input.json")
        if tpl_path is None:
            raise FileNotFoundError(
                "Template not found: templates/deepmd/input.json.template. "
                "Set CATFLOW_USE_TEMPLATE=0 to use dpgen, or provide the template file."
            )

        for model_idx in range(numb_models):
            model_dir = train_dir / f"{str(model_idx).zfill(3)}"
            model_dir.mkdir(parents=True, exist_ok=True)

            render_template(tpl_path, {
                "TYPE_MAP": type_map_str,
                "SEL": sel_str,
                "SEED": str(seeds[model_idx]),
                "STEPS": str(n_steps),
                "DECAY_STEPS": str(decay_steps),
                "DP_DATASET": dataset_str,
            }, str(model_dir / "input.json"))

        logger.info(f"[TMPL] Generated {numb_models} training tasks in {train_dir}")
        return 0

    def run(self):
        """Run training via oh-my-batch (omb) if available, fallback to dpgen."""
        if self._use_omb():
            self._omb_run_train()
        else:
            _require_dpgen().generator.run.run_train(
                self.step_code, self.params, self.machine)

    def _use_omb(self) -> bool:
        """Check if oh-my-batch should be used for this step."""
        return os.environ.get("CATFLOW_USE_OMB", "0") == "1"

    def _omb_run_train(self):
        """Run DeePMD training via omb combo + batch + job."""
        work_dir = Path.cwd()
        iter_dir = work_dir / f"iter.{str(self.step_code).zfill(6)}"
        train_dir = iter_dir / "00.train"
        train_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[OMB] Preparing training tasks in {train_dir}")

        # Generate input files (use template if available, else dpgen)
        if self._use_template():
            self._make_template()
        else:
            _require_dpgen().generator.run.make_train(
                self.step_code, self.params, self.machine
            )

        # Generate and submit via omb bash script
        try:
            n_models = self.params.get("numb_models", 4)
            n_steps = self.params.get("default_training_param", {}).get(
                "training", {}).get("numb_steps", 400000)
        except Exception:
            n_models = 4
            n_steps = 400000

        script = TeslaBatchScript(
            machine=MachineConfig(
                name="tesla_train",
                scheduler="slurm",
                resources=SbatchResources(
                    partition="gpu",
                    node_count=1,
                    cpu_per_node=4,
                    gpu_per_node=1,
                ),
            )
        )
        script.train_stage(
            work_dir=str(work_dir),
            model_count=n_models,
            train_steps=n_steps,
            concurrency=4,
        )
        script_path = script.write(str(train_dir / "run_omb_train.sh"))
        logger.info(f"[OMB] Training script written: {script_path}")

        executor = self._build_workflow_executor(str(work_dir))
        result = executor.run_script(script_path)
        if result.returncode != 0:
            raise RuntimeError(f"OMB training failed: {result.stderr}")

    def run_dpgen(self):
        """Legacy dpgen-based submission."""
        from dpgen.generator.run import run_train
        return run_train(self.step_code, self.params, self.machine)

    def post(self):
        if self._use_template():
            from catflow.tasker.collectors.train import collect_train_results
            return collect_train_results(self.step_code)
        _require_dpgen().generator.run.post_train(
            self.step_code, self.params, self.machine)


class DPExploration(TeslaWorkStep):
    def make(self):
        if self._use_template():
            return self._make_template()
        _require_dpgen().generator.run.make_model_devi(
            self.step_code, self.params, self.machine)

    def _make_template(self):
        """Generate LAMMPS exploration input files from templates."""
        import dpdata

        iter_idx = self.step_code
        iter_dir = Path(f"iter.{str(iter_idx).zfill(6)}")
        exp_dir = iter_dir / "01.model_devi"
        exp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[TMPL] Generating exploration tasks in {exp_dir}")

        # --- symlink model graphs from training ---
        train_dir = iter_dir / "00.train"
        model_files = sorted(glob(str(train_dir / "*/frozen_model.pb")))
        if not model_files:
            model_files = sorted(glob(str(train_dir / "*/graph.pb")))
        for i, mf in enumerate(model_files):
            target = exp_dir / f"graph.{str(i).zfill(3)}.pb"
            if not target.exists():
                target.symlink_to(Path(mf).resolve())

        # --- mass_map from params ---
        mass_map = self.params.get("mass_map", [])
        mass_map_str = " ".join(str(m) for m in mass_map)

        # --- conf setup (from init structures) ---
        sys_configs = self.params.get("sys_configs", [])
        sys_prefix = self.params.get("sys_configs_prefix", ".")
        conf_dir = exp_dir / "confs"
        conf_dir.mkdir(exist_ok=True)

        for sys_idx, cfgs in enumerate(sys_configs):
            for conf_idx, cfg_path in enumerate(cfgs):
                full_path = Path(sys_prefix) / cfg_path
                if full_path.exists():
                    poscar_link = conf_dir / f"{sys_idx}.{conf_idx}.poscar"
                    if not poscar_link.exists():
                        poscar_link.symlink_to(full_path.resolve())
                    # Convert to LAMMPS data
                    lmp_path = conf_dir / f"{sys_idx}.{conf_idx}.lmp"
                    if not lmp_path.exists():
                        try:
                            sys_data = dpdata.System(str(full_path), fmt="vasp/poscar")
                            sys_data.to_lammps_lmp(str(lmp_path))
                        except Exception:
                            pass

        # --- job config from params ---
        model_devi_jobs = self.params.get("model_devi_jobs", [])
        cur_job = model_devi_jobs[iter_idx] if iter_idx < len(model_devi_jobs) else {}
        rev_mat = cur_job.get("rev_mat", {})
        lmp_conf = rev_mat.get("lmp", {})
        temps = lmp_conf.get("V_TEMP", [300])
        pres = lmp_conf.get("V_PRES", [1])
        nsteps = lmp_conf.get("V_NSTEPS", [10000])
        dt = lmp_conf.get("V_DT", [0.002])

        model_list = " ".join(
            f"../graph.{str(i).zfill(3)}.pb" for i in range(len(model_files))
        )
        sys_idx_list = cur_job.get("sys_idx", list(range(len(sys_configs))))

        # --- generate LAMMPS input template ---
        tpl_path = find_template("lammps/explore.in")
        if tpl_path is None:
            raise FileNotFoundError("Template not found: templates/lammps/explore.in.template")

        task_idx = 0
        for sidx in sys_idx_list:
            for temp in temps:
                task_dir = exp_dir / f"task.{sidx}.{task_idx}"
                task_dir.mkdir(parents=True, exist_ok=True)

                # Link conf
                conf_link = task_dir / "conf.lmp"
                conf_src = conf_dir / f"{sidx}.0.lmp"
                if conf_src.exists() and not conf_link.exists():
                    conf_link.symlink_to(conf_src)

                seed = np.random.randint(100000, 999999)

                render_template(tpl_path, {
                    "NSTEPS": str(nsteps[0] if isinstance(nsteps, list) else nsteps),
                    "TEMP": str(temp),
                    "SEED": str(seed),
                    "DP_MODELS": model_list,
                    "MASS_MAP": mass_map_str,
                    "DT": str(dt[0] if isinstance(dt, list) else dt),
                }, str(task_dir / "input.lammps"))

                # Write job.json
                with open(task_dir / "job.json", "w") as f:
                    json.dump({"sys_idx": sidx, "temp": temp, "pres": pres}, f)

                task_idx += 1

        logger.info(f"[TMPL] Generated {task_idx} exploration tasks in {exp_dir}")
        return 0

    def run(self):
        """Run exploration via oh-my-batch (omb) if available, fallback to dpgen."""
        if self._use_omb():
            self._omb_run_explore()
        else:
            _require_dpgen().generator.run.run_model_devi(
                self.step_code, self.params, self.machine)

    def _use_omb(self) -> bool:
        return os.environ.get("CATFLOW_USE_OMB", "0") == "1"

    def _omb_run_explore(self):
        """Run LAMMPS exploration via omb combo + batch + job."""
        work_dir = Path.cwd()
        iter_dir = work_dir / f"iter.{str(self.step_code).zfill(6)}"
        explore_dir = iter_dir / "01.model_devi"
        explore_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[OMB] Preparing exploration tasks in {explore_dir}")

        # Generate input files (use template if available, else dpgen)
        if self._use_template():
            self._make_template()
        else:
            _require_dpgen().generator.run.make_model_devi(
                self.step_code, self.params, self.machine
            )

        script = TeslaBatchScript(
            machine=MachineConfig(
                name="tesla_explore",
                scheduler="slurm",
                resources=SbatchResources(
                    partition="gpu",
                    node_count=1,
                    cpu_per_node=4,
                    gpu_per_node=0,
                ),
            )
        )
        script.explore_stage(
            work_dir=str(work_dir),
            iter_name=str(self.step_code).zfill(6),
            concurrency=5,
        )
        script_path = script.write(str(explore_dir / "run_omb_explore.sh"))
        logger.info(f"[OMB] Exploration script written: {script_path}")

        executor = self._build_workflow_executor(str(work_dir))
        result = executor.run_script(script_path)
        if result.returncode != 0:
            raise RuntimeError(f"OMB exploration failed: {result.stderr}")

    def post(self):
        if self._use_template():
            from catflow.tasker.collectors.exploration import (
                collect_exploration_results, generate_shuffled_stats
            )
            results = collect_exploration_results(self.step_code)
            generate_shuffled_stats(
                results,
                f_trust_lo=self.params.get("model_devi_f_trust_lo", 0.1),
                f_trust_hi=self.params.get("model_devi_f_trust_hi", 0.3),
                iter_index=self.step_code,
            )
            return results
        _require_dpgen().generator.run.post_model_devi(
            self.step_code, self.params, self.machine)


class FPCalculation(TeslaWorkStep):
    def make(self):
        if self._use_template():
            return self._make_template()
        _require_dpgen().generator.run.make_fp(
            self.step_code, self.params, self.machine)

    def _make_template(self):
        """Generate FP labeling input files from templates."""
        import dpdata

        iter_idx = self.step_code
        iter_dir = Path(f"iter.{str(iter_idx).zfill(6)}")
        fp_dir = iter_dir / "02.fp"
        fp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[TMPL] Generating FP tasks in {fp_dir}")

        fp_style = self.params.get("fp_style", "vasp")
        prev_exp_dir = iter_dir / "01.model_devi"
        sys_idx_list = self.params.get("sys_configs", [])
        task_idx = 0

        # --- select candidate frames from model_devi trajectories ---
        for sys_idx in range(len(sys_idx_list)):
            traj_files = sorted(glob(str(prev_exp_dir / f"task.{sys_idx}.*/traj/*.lammpstrj")))
            candidate_out = []
            accurate_out = []
            failed_out = []

            for traj_file in traj_files:
                task_name = Path(traj_file).parent.parent.name
                try:
                    frames = dpdata.System(traj_file, fmt="lammps/dump")
                    # Select middle frames (skip initial equilibration)
                    start = max(0, len(frames) // 10)
                    for frame_idx in range(start, len(frames), max(1, len(frames) // 20)):
                        task_fp_dir = fp_dir / f"task.{sys_idx}.{task_idx}"
                        task_fp_dir.mkdir(parents=True, exist_ok=True)

                        # Write POSCAR
                        frame = frames[frame_idx]
                        frame.to_vasp_poscar(str(task_fp_dir / "POSCAR"))

                        # For CP2K: write coord.xyz
                        if fp_style == "cp2k":
                            frame.to_cp2k_xyz(str(task_fp_dir / "coord.xyz"))

                        candidate_out.append(f"{task_name} {frame_idx}")
                        task_idx += 1
                except Exception as e:
                    logger.warning(f"Failed to process {traj_file}: {e}")

            # Write shuffled output files
            if candidate_out:
                with open(fp_dir / f"candidate.shuffled.{str(sys_idx).zfill(3)}.out", "w") as f:
                    f.write("\n".join(candidate_out) + "\n")

        # --- generate input files per FP style ---
        if fp_style == "vasp":
            self._make_fp_vasp_inputs(fp_dir, iter_idx)
        elif fp_style == "cp2k":
            self._make_fp_cp2k_inputs(fp_dir)

        logger.info(f"[TMPL] Generated {task_idx} FP tasks in {fp_dir}")
        return 0

    def _make_fp_vasp_inputs(self, fp_dir: Path, iter_idx: int):
        """Generate VASP INCAR/POTCAR for FP tasks."""
        fp_params = self.params.get("fp_params", {})
        user_fp_params = self.params.get("user_fp_params", {})
        fp_incar = fp_params.get("fp_incar", {}) or user_fp_params

        task_dirs = sorted(glob(str(fp_dir / "task.*")))
        for task_dir in task_dirs:
            td = Path(task_dir)
            # INCAR: write from fp_params
            if fp_incar:
                shutil.copy(fp_incar, td / "INCAR")
            # POTCAR: link from potential directory
            potcar_dir = fp_params.get("potcar_prefix", ".")
            potcar_map = fp_params.get("potcar_map", {})
            # Generate POTCAR for each element
            potcar_path = td / "POTCAR"
            if not potcar_path.exists():
                try:
                    from ase.io import read as ase_read
                    atoms = ase_read(td / "POSCAR")
                    symbols = set(atoms.get_chemical_symbols())
                    with open(potcar_path, "w") as potcar_out:
                        for sym in symbols:
                            potcar_file = Path(potcar_dir) / potcar_map.get(sym, f"POTCAR.{sym}")
                            if potcar_file.exists():
                                potcar_out.write(potcar_file.read_text())
                except Exception:
                    pass
            # KPOINTS
            kspacing = fp_params.get("kspacing", 0.2)
            with open(td / "KPOINTS", "w") as f:
                f.write(f"KSPACING = {kspacing}\n")

    def _make_fp_cp2k_inputs(self, fp_dir: Path):
        """Generate CP2K input files for FP tasks using templates."""
        tpl_path = find_template("cp2k/fp.inp")
        if tpl_path is None:
            logger.warning("CP2K template not found, using default input generation")
            return

        fp_params = self.params.get("fp_params", {})

        task_dirs = sorted(glob(str(fp_dir / "task.*")))
        for task_dir in task_dirs:
            td = Path(task_dir)
            render_template(tpl_path, {
                "PROJECT": "cp2k_fp",
                "BASIS_FILE": fp_params.get("basis_file", "BASIS_MOLOPT"),
                "POTENTIAL_FILE": fp_params.get("potential_file", "POTENTIAL"),
                "CUTOFF": str(fp_params.get("cutoff", 400)),
                "XC_FUNCTIONAL": fp_params.get("xc_functional", "PBE"),
                "CELL_PARAMS": "",
                "COORD_FILE": "",
                "COORD_FORMAT": "XYZ",
                "COORD_FILE_NAME": "coord.xyz",
            }, str(td / "input.inp"))

    def run(self):
        """Run FP labeling via oh-my-batch (omb) if available, fallback to dpgen."""
        if self._use_omb():
            self._omb_run_fp()
        else:
            _require_dpgen().generator.run.run_fp(
                self.step_code, self.params, self.machine)

    def _use_omb(self) -> bool:
        return os.environ.get("CATFLOW_USE_OMB", "0") == "1"

    def _omb_run_fp(self):
        """Run CP2K/VASP labeling via omb batch + job."""
        work_dir = Path.cwd()
        iter_dir = work_dir / f"iter.{str(self.step_code).zfill(6)}"
        label_dir = iter_dir / "02.fp"
        label_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[OMB] Preparing labeling tasks in {label_dir}")

        # Generate input files (use template if available, else dpgen)
        if self._use_template():
            self._make_template()
        else:
            _require_dpgen().generator.run.make_fp(
                self.step_code, self.params, self.machine
            )

        # Detect software (CP2K or VASP)
        fp_style = self.params.get("fp_style", "cp2k")

        script = TeslaBatchScript(
            machine=MachineConfig(
                name="tesla_label",
                scheduler="slurm",
                resources=SbatchResources(
                    partition="cpu",
                    node_count=1,
                    cpu_per_node=8,
                    gpu_per_node=0,
                ),
            )
        )
        script.labeling_stage(
            work_dir=str(work_dir),
            iter_name=str(self.step_code).zfill(6),
            software=fp_style,
            concurrency=5,
        )
        script_path = script.write(str(label_dir / "run_omb_label.sh"))
        logger.info(f"[OMB] Labeling script written: {script_path}")

        executor = self._build_workflow_executor(str(work_dir))
        result = executor.run_script(script_path)
        if result.returncode != 0:
            raise RuntimeError(f"OMB labeling failed: {result.stderr}")

    def post(self):
        if self._use_template():
            from catflow.tasker.collectors.labeling import collect_labeling_results
            return collect_labeling_results(self.step_code)
        _require_dpgen().generator.run.post_fp(
            self.step_code, self.params)


class CLWorkflow(object):
    def __init__(self, param_file, machine_pool, conf_file=None):
        """initialize the concurrent learning workflow

        Parameters
        ----------
        param_file : a file containing all the parameters for workflow runs
        machine_pool : a file with machine for each step to use
        """
        self.param_file = Path(param_file).resolve()
        self.machine_pool = Path(machine_pool).resolve()
        if conf_file is not None:
            workflow_settings = self.set_workflow(conf_file)
            self.workflow_settings = workflow_settings
        else:
            self.workflow_settings = {}
        for key in workflow_settings.keys():
            setattr(self, key, workflow_settings[key])
        self.stage = 0
        self.step = 0
        self.params = self.get_data(self.param_file)
        self.updater = CLWorkflowUpdater

    @staticmethod
    def set_workflow(conf_file):
        with open(conf_file) as f:
            conf = load(f, Loader=SafeLoader)
        return conf

    @staticmethod
    def get_data(json_file):
        with open(json_file, 'r', encoding='utf-8') as fp:
            return json.load(fp)

    @staticmethod
    def record_stage(record_file_path, stage, step):
        record_array = np.array([stage, step])
        np.savetxt(record_file_path, record_array, fmt="%d")

    @property
    def machine(self):
        """Load and parse machine configuration.

        Reads the machine pool JSON and returns a standardized dict.
        Replaces dpgen.remote.decide_machine.convert_mdata.
        """
        mdata = self.get_data(self.machine_pool)
        # If dpgen is not installed, provide a simple identity mapping
        try:
            from dpgen.remote.decide_machine import convert_mdata
            return convert_mdata(mdata)
        except ImportError:
            # Fallback: convert to dpgen-compatible format ourselves
            converted = {}
            for task_type in ["train", "model_devi", "fp"]:
                task_key = f"{task_type}_machine"
                if task_type in mdata:
                    converted[task_key] = mdata[task_type].get("machine", mdata[task_type])
                elif "machine" in mdata.get(task_type, {}):
                    converted[task_key] = mdata[task_type]["machine"]
            return converted

    @property
    def work_path(self):
        return Path.cwd()

    @property
    def main_step(self):
        return int(self.step) // 3

    @property
    def real_step(self):
        return int(self.step) % 3

    @property
    def main_step_dict(self):
        return {
            0: DPTrain,
            1: DPExploration,
            2: FPCalculation,
        }

    def get_work_step(self, step_code):
        main_step_dict = self.main_step_dict
        step = main_step_dict.get(step_code)
        return step

    def read_record(self, record="miko.record"):
        record_file_path = Path(record).resolve()
        try:
            stage_rec = np.loadtxt(record_file_path, dtype=int)
            self.stage = stage_rec[0]
            self.step = stage_rec[1]
            logger.info("continue from stage {0} step {1}".format(
                self.stage, self.step))
        except FileNotFoundError or NameError:
            logger.debug("record file not found")
            logger.debug("creating record file {0}".format(record))
            self.record_stage(record_file_path, self.stage, self.step)

    def run_step(self):
        logger.info("now stage: {0}, step: {1}".format(
            self.stage, self.step))
        logger.info("now main step: {}".format(self.main_step))
        logger.info("now real step: {}".format(self.real_step))
        stage_class = self.get_work_step(self.main_step)
        stage_task = stage_class(self.params, self.stage, self.machine)
        step_task = stage_task.sub_step_dict.get(self.real_step)
        step_task()

        if self.step != 8:
            self.step += 1
        else:
            self.stage += 1
            self.step = 0

    def run_loop(self, record="miko.record"):
        record_file_path = Path(record).resolve()
        self.read_record(record_file_path)
        while self.check_converge():
            self.run_step()
            self.record_stage(record_file_path, self.stage, self.step)
        self.append_tasks(record=record)

    def check_converge(self):
        """check if model deviation converged after CL runs.

        Returns:
            Bool: whether continue or not.
        """
        if self.main_step == 1:
            if self.real_step == 0:
                model_devi_jobs = self.params.get('model_devi_jobs')
                try:
                    last_model_devi_job = model_devi_jobs[-1]
                except IndexError:
                    last_model_devi_job = {}
                last_sys_idx = last_model_devi_job.get('sys_idx', [])
                if len(last_sys_idx) == 0:
                    logger.info('Empty model_devi_job found')
                    self.update_params()
                    return True
                else:
                    conv_flags = np.zeros_like(last_sys_idx, dtype=int)
                    logger.info(
                            f'Checking convergency for iteration {self.stage}')
                    for i, idx in enumerate(last_sys_idx):
                        accu_ratio = self._check_index_converge(idx)
                        logger.info(
                            f'idx {idx} reach accuracy ratio: {round(accu_ratio, 3)}')
                        if accu_ratio >= 0.97:
                            conv_flags[i] = 1
                    if 0 in conv_flags:
                        # not all accu, update params
                        logger.info('Not all idxs reach 97% accuracy')
                        logger.info('Continue training process')
                        self.update_params()
                        return True
                    elif self.stage <= self.workflow_settings.get('start_from_iter', 0):
                        # prevent not starting
                        logger.info('This is the first iteration')
                        self.update_params()
                        return True
                    else:
                        unfinished_exploration = last_model_devi_job.get("unfinished_exploration", False)
                        if unfinished_exploration:
                            # not all reaction coordinate covered
                            logger.info("Exploration not ending, try again.")
                            self.update_params()
                            return True
                        else:
                            # all accu, finish exploration
                            return False
            else:
                return True
        else:
            return True

    def append_tasks(self, record=None):
        self.long_train_process()

    def long_train_process(self):
        logger.info('Model accuracy converged.')
        if self.params.get("auto_long_train", False) == True:
            logger.info('Long train task starts.')
            long_train_task = LongTrain(self)
            long_train_task.run_long_train()
    
    def update_params(self):
        if callable(self.updater):
            updater = self.updater(self)
            updater.model_devi_job_generator()
        self._set_cur_trust_level()
        self.render_params()

    def render_params(self, param_file=None):
        """render the params.json file for human reading and normal dpgen runs.

        Args:
            param_file (_type_, optional): _description_. Defaults to None.
        """
        params = self.params
        if param_file == None:
            param_file = self.param_file
            shutil.copyfile(
                self.param_file,
                self.param_file.parent / f"params_backup_{self.stage}.json"
            )
        with open(param_file, "w", encoding='utf-8') as f:
            json.dump(params, f, indent=4)

    def _set_cur_trust_level(self):
        f_trust_lo, f_trust_hi = self.guess_trust_level()
        cur_job = self.params['model_devi_jobs'][self.stage]
        cur_job["model_devi_f_trust_lo"] = f_trust_lo
        cur_job["model_devi_f_trust_hi"] = f_trust_hi

    def guess_trust_level(self, stage: int = None) -> float:
        """guess trust level from training step each iteration

        Args:
            stage (int, optional): The iteration to guess trust level from. Default: current stage.

        Returns:
            float: lower and higher limitation of force deviation trust level
        """

        if stage is None:
            stage = self.stage
        mean_l2_error = 0.20

        try:
            training_l2_error = np.loadtxt(
                Path(f"iter.{str(stage).zfill(6)}/00.train/000/lcurve.out").resolve(), usecols=3)
            mean_l2_error = np.mean(training_l2_error[-int(len(training_l2_error)/10):])
            logger.info("Use values guessed from training.")
        except FileNotFoundError:
            logger.info("Training task not found. Use default values.")

        logger.info(
            f"f_trust_lo: {round(mean_l2_error * 0.9, 2)}, f_trust_hi: {round(mean_l2_error * 3.0, 2)}")
        return round(mean_l2_error * 0.9, 2), round(mean_l2_error * 3.0, 2)

    @staticmethod
    def _trust_limitation_check(sys_idx, lim):
        if isinstance(lim, list):
            sys_lim = lim[sys_idx]
        elif isinstance(lim, dict):
            sys_lim = lim[str(sys_idx)]
        else:
            sys_lim = lim
        return sys_lim

    def get_trust_level(self, param_type, sys_idx):
        """get trust level for 
        Args:
            param_type (_type_): _description_
            sys_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        param_detail = self.params['model_devi_jobs'][self.stage -
                                                      1].get(param_type)
        if param_detail is None:
            param_detail = self.params.get(param_type)
        return self._trust_limitation_check(sys_idx, param_detail)

    def _check_index_converge(self, index):
        """count number of frames of each type through history fp

        Args:
            index (int): sys_idx of file

        Returns:
            float: ratio of accurate frames
        """
        accu_count = self._check_shuffled_log(
            f"rest_accurate.shuffled.{str(index).zfill(3)}.out")
        failed_count = self._check_shuffled_log(
            f"rest_failed.shuffled.{str(index).zfill(3)}.out")
        candidate_count = self._check_shuffled_log(
            f"candidate.shuffled.{str(index).zfill(3)}.out")
        all_count = accu_count + failed_count + candidate_count
        try:
            accu_ratio = accu_count / all_count
        except ZeroDivisionError:
            logger.info("no exploration task for idx {}".format(index))
            accu_ratio = 1.
        return accu_ratio

    def _check_shuffled_log(self, shuffled_logs):
        shuffled_logs = glob(
            os.path.join(
                f'iter.{str(self.stage - 1).zfill(6)}',
                '02.fp',
                shuffled_logs
            )
        )
        log_count = 0
        for log_file in shuffled_logs:
            with open(log_file) as f:
                log_count += count_lines(f)
        return log_count


class CLWorkflowUpdater:
    """update the params during each iteration loop
    """
    def __init__(self, wf: CLWorkflow):
        self.workflow = wf
        self.template_key = "job_template"

    def model_devi_job_generator(self):
        while len(self.workflow.params['model_devi_jobs']) < self.workflow.stage + 1:
            logger.debug(f"Add new model_devi_job.")
            try:
                last_job = self.workflow.params['model_devi_jobs'][-1]
            except IndexError:
                last_job = self.get_init()
            self.workflow.params['model_devi_jobs'].append(last_job)
        cur_job = self.workflow.params['model_devi_jobs'][self.workflow.stage]
        self._new_template_generator(cur_job)

    def get_init(self):
        init_job = self.workflow.workflow_settings.get("job_template")
        return init_job

    def _new_template_generator(self, cur_job):
        cur_job["_idx"] = str(self.workflow.stage)


class ClusterReactionWorkflow(CLWorkflow):
    def __init__(self, param_file, machine_pool, conf_file):
        super().__init__(param_file, machine_pool, conf_file)
        self.updater = ClusterReactionUpdater

    def append_tasks(self, record=None):
        logger.info("Check if continue to run metadynamics exploration")
        if self.workflow_settings.get("append_metad_flow"):
            metad_workflow = MetadynReactionWorkflow(
                self.param_file, 
                self.machine_pool, 
                self.conf_file
            )
            metad_workflow.workflow_settings['start_from_iter'] = self.stage
            metad_workflow.run_loop(record=record)
        else:
            self.long_train_process()


class ClusterReactionUpdater:
    """update the params during each iteration loop
    """

    def __init__(self, wf: ClusterReactionWorkflow):
        self.workflow = wf

    def model_devi_job_generator(self):
        while len(self.workflow.params['model_devi_jobs']) < self.workflow.stage + 1:
            logger.debug(f"Add new model_devi_job.")
            self.workflow.params['model_devi_jobs'].append(
                self.workflow.params['model_devi_jobs'][-1])
        cur_job = self.workflow.params['model_devi_jobs'][self.workflow.stage]
        self._new_template_generator(cur_job)

    @property
    def exploration_step(self):
        try:
            return self.workflow.exploration_step
        except AttributeError:
            return 0.1

    @property
    def exploration_track(self):
        is_coord = self.workflow.IS['coordination']
        fs_coord = self.workflow.FS['coordination']
        full_track = np.arange(is_coord, fs_coord +
                               self.exploration_step, self.exploration_step)
        return full_track

    def _new_template_generator(self, cur_job):
        cur_job["_idx"] = str(self.workflow.stage)
        # get IS and FS sys_idx and coordination from self.workflow.workflow_settings
        is_sys_idx, is_coord = self._get_cv_setting('IS')
        fs_sys_idx, fs_coord = self._get_cv_setting('FS')
        if 'TS' in self.workflow.workflow_settings.keys():
            ts_sys_idx, ts_coord = self._get_cv_setting('TS')

        exploration_track = self.exploration_track
        exploration_step = self.exploration_step

        logger.info(f"exploration step: {exploration_step}")
        logger.info(f"exploration track: {exploration_track}")

        ts_flag = False
        if 'TS' in self.workflow.workflow_settings.keys():
            ts_flag = True
            ts_coord = self.workflow.TS['coordination']

        add_new_flag = 0
        if ts_flag == False:
            center_idx = int(len(exploration_track) / 2 - 1)
            if cur_job.get("sys_rev_mat", {}) == {}:
                cur_job["sys_rev_mat"] = {}
                try:
                    cur_job["sys_idx"].append(is_sys_idx)
                except KeyError:
                    cur_job["sys_idx"] = []
                    cur_job["sys_idx"].append(is_sys_idx)
                cur_job["sys_rev_mat"][str(is_sys_idx)] = {
                    "lmp": {
                        "V_DIS1": [round(is_coord, 3)],
                        "V_DIS2": [round(is_coord, 3), round(is_coord + 2 * exploration_step, 3)],
                        "V_FORCE": [10],
                    },
                    "_type": "IS"
                }
                cur_job["sys_idx"].append(fs_sys_idx)
                cur_job["sys_rev_mat"][str(fs_sys_idx)] = {
                    "lmp": {
                        "V_DIS1": [round(fs_coord, 3)],
                        "V_DIS2": [round(fs_coord, 3), round(fs_coord - 2 * exploration_step, 3)],
                        "V_FORCE": [10]
                    },
                    "_type": "FS"
                }
                add_new_flag += 2
                cur_job["unfinished_exploration"] = True
            else:
                for key in cur_job['sys_rev_mat'].keys():
                    cur_job['sys_rev_mat'][key]['lmp']['V_DIS2'] = cur_job['sys_rev_mat'][key]['lmp']['V_DIS1']

                task_list, distances = self._distances()

                is_coords = [i['lmp']['V_DIS1'][0]
                             for i in cur_job['sys_rev_mat'].values() if i.get('_type') == 'IS']
                logger.info(f"explored IS coords: {is_coords}")
                
                unfinished_exploration = False
                for new_coord in [
                    round(max(is_coords) + exploration_step, 3),
                    round(max(is_coords) + 2 * exploration_step, 3)
                ]:
                    if new_coord < self.exploration_track[center_idx]:
                        unfinished_exploration = True
                        is_sys_idx = self._pick_new_structure(new_coord, task_list, distances)
                        logger.debug(f"add new sys_idx: {is_sys_idx}")
                        if is_sys_idx is not None:
                            logger.info(
                                f"add new exploration coord: {new_coord}")
                            cur_job["sys_idx"].append(is_sys_idx)
                            cur_job["sys_rev_mat"][str(is_sys_idx)] = {
                                "lmp": {
                                    "V_DIS1": [round(new_coord, 3)],
                                    "V_DIS2": [round(new_coord, 3), round(new_coord + 2 * exploration_step, 3)],
                                    "V_FORCE": [10]
                                },
                                "_type": "IS"
                            }
                            add_new_flag += 1

                fs_coords = [i['lmp']['V_DIS1'][0]
                             for i in cur_job['sys_rev_mat'].values() if i.get('_type') == 'FS']
                logger.info(f"explored FS coords: {fs_coords}")

                for new_coord in [
                    round(min(fs_coords) - exploration_step, 3), 
                    round(min(fs_coords) - 2 * exploration_step, 3)
                ]:
                    if new_coord >= self.exploration_track[center_idx]:
                        unfinished_exploration = True
                        fs_sys_idx = self._pick_new_structure(new_coord, task_list, distances)
                        if fs_sys_idx is not None:
                            logger.info(
                                f"add new exploration coords: {new_coord}")
                            cur_job["sys_idx"].append(fs_sys_idx)
                            cur_job["sys_rev_mat"][str(fs_sys_idx)] = {
                                "lmp": {
                                    "V_DIS1": [round(new_coord, 3)],
                                    "V_DIS2": [round(new_coord, 3), round(new_coord - 2 * exploration_step, 3)],
                                    "V_FORCE": [10]
                                },
                                "_type": "FS"
                            }
                            add_new_flag += 1
                cur_job["unfinished_exploration"] = unfinished_exploration
        
        #TODO: generator for TS

        if add_new_flag == 0:
            cur_job["rev_mat"]["lmp"]["V_NSTEPS"] = [i * 2 for i in cur_job["rev_mat"]["lmp"]["V_NSTEPS"]]

    def _distances(self):
        task_list = []
        start_iter = self.workflow.workflow_settings.get('start_xfrom_iter', 0)
        for i in range(start_iter, self.workflow.stage):
            task_list += sorted((self.workflow.work_path / f"iter.{str(i).zfill(6)}").glob(
                '02.fp/task.*.*/POSCAR'))
        distances = []
        for i in task_list:
            _s = read(i)
            distances.append(_s.get_distance(
                *self.workflow.reaction_atoms_pair))
        return task_list, distances

    def _pick_new_structure(self, new_start_coord, task_list, distances):
        try:
            new_idx = np.where((np.array(distances) >= new_start_coord - 0.05)
                               & (np.array(distances) < new_start_coord + 0.05))[0][0]
        except IndexError:
            logger.info("New structure not found!")
            return None
        new_structure_path = Path(task_list[new_idx])
        new_sys_idx = self._render_new_system(new_structure_path)
        return new_sys_idx

    def _render_new_system(self, new_structure_path: Path):
        sys_configs_prefix = self.workflow.params["sys_configs_prefix"]
        new_sys_item = [
            str(new_structure_path.relative_to(sys_configs_prefix))]
        self.workflow.params["sys_configs"].append(new_sys_item)
        if self.workflow.params.get("sys_batch_size"):
            self.workflow.params["sys_batch_size"].append("auto")
        return len(self.workflow.params["sys_configs"]) - 1

    def _get_cv_setting(self, cv_type):
        sys_idx = self.workflow.workflow_settings[cv_type]['sys_idx']
        coord = self.workflow.workflow_settings[cv_type]['coordination']
        return sys_idx, coord


class MetadynReactionWorkflow(CLWorkflow):
    """Metadynamics workflow
    """
    def __init__(self, param_file, machine_pool, conf_file):
        super().__init__(param_file, machine_pool, conf_file)
        self.updater = MetaDynReactionUpdater


class MetaDynReactionUpdater:
    """update the params during each iteration loop
    """
    def __init__(self, wf: CLWorkflow):
        self.workflow = wf
        self.template_key = "metad_job_template"

    def model_devi_job_generator(self):
        while len(self.workflow.params['model_devi_jobs']) < self.workflow.stage + 1:
            logger.debug(f"Add new model_devi_job.")
            try:
                last_job = self.workflow.params['model_devi_jobs'][-1]
            except IndexError:
                last_job = self.get_init()
            self.workflow.params['model_devi_jobs'].append(last_job)
        if self.workflow.stage >= self.workflow.workflow_settings.get("start_from_iter", 0):
            self.workflow.params['model_devi_jobs'][-1] = self.get_init()
        cur_job = self.workflow.params['model_devi_jobs'][self.workflow.stage]
        self._new_template_generator(cur_job)

    def get_init(self):
        init_job = self.workflow.workflow_settings.get(self.template_key)
        return init_job

    def _new_template_generator(self, cur_job):
        cur_job["_idx"] = str(self.workflow.stage)


class LongTrain:
    """Final step for CLWorkFlow"""
    def __init__(self, wf: CLWorkflow) -> None:
        self.workflow = wf

    def update_params(self):
        decay_steps = self.workflow.params["default_training_param"]["learning_rate"]["decay_steps"] 
        self.workflow.params["default_training_param"]["learning_rate"]["decay_steps"] = int(decay_steps) * 10
        numb_steps = self.workflow.params["default_training_param"]["training"]["numb_steps"]
        self.workflow.params["default_training_param"]["training"]["numb_steps"] = numb_steps * 10

    def run_long_train(self):
        self.update_params()

        train_task = DPTrain(self.workflow.params, self.workflow.stage, self.workflow.machine)
        for i in range(3):
            train_sub_task = train_task.sub_step_dict.get(i)
            train_sub_task()
