import json
import os
import shutil
from glob import glob
from pathlib import Path

import dpdata
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from catflow.utils.log_factory import logger
from catflow.analyzer.tesla.dpgen.task import DPTask
from catflow.tasker.resources.submit import JobFactory
from catflow.tasker.resources.config import MachineConfig, SbatchResources
from catflow.tasker.resources.script_gen import ScriptBuilder
from catflow.tasker.resources.workflow_executor import WorkflowExecutor
from catflow.utils.lammps import \
    convert_init_structures, check_keywords, \
    parse_template, substitute_keywords

class DPCheck:
    def __init__(self, dp_task: DPTask):
        self.task = dp_task

    def md_single_task(
            self,
            work_path,
            model_path,
            machine_name,
            resource_dict,
            numb_models=4,
            **kwargs
    ):
        """Submit your own md task with the help of DPDispatcher.

        Parameters
        ----------
        work_path : str
            The dir contains your md tasks.
        model_path : str
            The path of models contained for calculation.
        machine_name : str
            machine name to use
        resource_dict : dict
            resource dict
        numb_models : int, optional
            The number of models selected., by default 4

        Returns
        -------
        JobFactory
            To generate job to be submitted.
        """
        mdata = self.task.machine_data['model_devi'][0]
        folder_list = kwargs.get('folder_list', ["task.*"])
        all_task = []
        for i in folder_list:
            _task = glob(os.path.join(work_path, i))
            _task.sort()
            all_task += _task
        lmp_exec = mdata['command']
        command = lmp_exec + " -i input.lammps"
        run_tasks_ = all_task
        run_tasks = [os.path.basename(ii) for ii in run_tasks_]

        model_names = kwargs.get(
            'model_names', [f'graph.{str(i).zfill(3)}.pb' for i in range(numb_models)])
        for ii in model_names:
            if not os.path.exists(os.path.join(work_path, ii)):
                os.symlink(os.path.join(model_path, ii),
                           os.path.join(work_path, ii))
        forward_files = kwargs.get(
            'forward_files', ['conf.lmp', 'input.lammps', 'traj'])
        backward_files = kwargs.get(
            'backward_files', ['model_devi.out', 'model_devi.log', 'traj'])

        task_dict_list = [
            {
                "command": command,
                "task_work_path": task,
                "forward_files": forward_files,
                "backward_files": backward_files,
                "outlog": kwargs.get('outlog', 'model_devi.log'),
                "errlog": kwargs.get('errlog', 'model_devi.log'),
            } for task in run_tasks
        ]

        submission_dict = {
            "work_base": work_path,
            "forward_common_files": model_names,
            "backward_common_files": kwargs.get('backward_common_files', [])
        }
        job = JobFactory(task_dict_list, submission_dict, machine_name, resource_dict)
        return job.submission

    def md_single_task_omb(
            self,
            work_path,
            model_path,
            concurrency=4,
            lmp_command="lmp -i input.lammps",
            **kwargs
    ):
        """Submit LAMMPS MD tasks via oh-my-batch.

        Args:
            work_path: Directory containing task.* subdirectories.
            model_path: Path to DeepMD model files (graph.*.pb).
            concurrency: Number of parallel tasks per batch script.
            lmp_command: LAMMPS execution command.
            kwargs: Additional options (forward_files, backward_files, etc.)
        """
        from glob import glob
        work_path = Path(work_path).resolve()
        model_path = Path(model_path).resolve()

        # Symlink models
        model_files = sorted(glob(str(model_path / "graph*.pb")))
        for mf in model_files:
            target = work_path / Path(mf).name
            if not target.exists():
                target.symlink_to(mf)

        task_dirs = sorted(glob(str(work_path / "task.*")))
        if not task_dirs:
            logger.warning(f"No tasks found in {work_path}/task.*")
            return

        logger.info(f"[OMB] Found {len(task_dirs)} tasks for MD test")

        sb = ScriptBuilder()
        sb.add_sbatch_header(MachineConfig(
            name="dpcheck",
            scheduler=kwargs.get('scheduler', 'slurm'),
            resources=SbatchResources(
                partition=kwargs.get('partition', 'gpu'),
                node_count=1,
                cpu_per_node=kwargs.get('cpu_per_node', 4),
                gpu_per_node=kwargs.get('gpu_per_node', 1),
            ),
        ))

        sb.add_comment("MD test via oh-my-batch")
        batch_cmd = (
            f'omb batch add_work_dirs "{work_path}/task.*"'
            f' add_cmd "{lmp_command}"'
            f' make "{work_path}/md-{{i}}.slurm" --concurrency {concurrency}'
        )
        sb.add_body(batch_cmd)

        recovery = kwargs.get('recovery_file', 'md_test_recovery.json')
        sb.add_body(
            f'omb job slurm submit "{work_path}/md-*.slurm"'
            f' --max_tries 2 --wait --recovery "{work_path}/{recovery}"'
        )

        script_path = sb.write(str(work_path / "run_omb_md.sh"))
        executor = WorkflowExecutor.local(work_base=str(work_path))
        result = executor.run_script(script_path)
        if result.returncode != 0:
            raise RuntimeError(f"OMB MD test failed: {result.stderr}")
        logger.info("[OMB] MD test completed.")

    def train_model_test(
            self,
            machine_name,
            resource_dict,
            iteration=None,
            params=None,
            restart=False,
            **kwargs
    ):
        """Run lammps MD tests from trained models.

        Parameters
        ----------
        machine_name : str
            machine name
        resource_dict : dict
            resource name
        iteration : str, optional
            Select the iteration for training. 
            If not selected, the last iteration where training has been finished would be chosen.
        params : str, optional
            Necessary params for MD tests.
        kwargs : str, optional
            Other optional parameters.

        Returns
        -------

        """
        location = os.path.abspath(self.task.path)
        logger.info(f"Task path: {location}")

        if iteration is None:
            if self.task.step_code < 2:
                iteration = self.task.iteration - 1
            else:
                iteration = self.task.iteration
        n_iter = 'iter.' + str(iteration).zfill(6)
        model_path = os.path.join(location, n_iter, '00.train')
        test_path = os.path.join(location, n_iter, '04.model_test')

        if params is None:
            params = self.task.param_data

        if restart == True:
            logger.info("Restarting from old checkpoint")
        else:
            logger.info("Preparing MD input")
            template_base = kwargs.get('template_base', None)
            self._train_generate_md_test(
                params=params,
                work_path=test_path,
                model_path=model_path,
                template_base=template_base
            )

        if self.task.step_code < 6:
            md_iter = self.task.iteration - 1
        else:
            md_iter = self.task.iteration
        md_iter = 'iter.' + str(md_iter).zfill(6)

        logger.info("Task submitting")
        job = self.md_single_task(
            work_path=test_path,
            model_path=model_path,
            machine_name=machine_name,
            resource_dict=resource_dict,
            numb_models=self.task.param_data['numb_models'],
            forward_files=kwargs.get(
                "forward_files", ['conf.lmp', 'input.lammps']),
            backward_files=kwargs.get("backward_files",
                                      ['model_devi.out', 'md_test.log', 'md_test.err', 'dump.lammpstrj']),
            outlog=kwargs.get("outlog", 'md_test.log'),
            errlog=kwargs.get("errlog", 'md_test.err')
        )
        job.run_submission()
        logger.info("MD Test finished.")

    def _train_generate_md_test(self, params, work_path, model_path, template_base=None):
        cur_job = params['md_test']
        use_plm = params.get('model_devi_plumed', False)
        trj_freq = cur_job.get('traj_freq', False)

        if template_base is None:
            template_base = self.task.path

        lmp_templ = cur_job['template']['lmp']
        lmp_templ = os.path.abspath(os.path.join(template_base, lmp_templ))
        plm_templ = None

        if use_plm:
            plm_templ = cur_job['template']['plm']
            plm_templ = os.path.abspath(os.path.join(template_base, plm_templ))

        sys_idxs = cur_job.get('sys_idx')
        for sys_idx in sys_idxs:
            sys_init_stc = convert_init_structures(params, sys_idx)
            templ, rev_mat = parse_template(cur_job, sys_idx)
            for task_counter, task in enumerate(rev_mat):
                task_name = "task.%03d.%06d" % (sys_idx, task_counter)
                task_path = os.path.join(work_path, task_name)

                # create task path
                os.makedirs(task_path, exist_ok=True)
                shutil.copyfile(lmp_templ, os.path.join(
                    task_path, 'input.lammps'))
                model_list = glob(os.path.join(model_path, 'graph*pb'))
                model_list.sort()
                model_names = [os.path.basename(i) for i in model_list]
                task_model_list = []
                for jj in model_names:
                    task_model_list.append(
                        os.path.join('..', os.path.basename(jj)))

                # revise input of lammps
                with open(os.path.join(task_path, 'input.lammps')) as fp:
                    lmp_lines = fp.readlines()
                _dpmd_idx = check_keywords(lmp_lines, ['pair_style', 'deepmd'])
                graph_list = ' '.join(task_model_list)
                lmp_lines[_dpmd_idx] = \
                    f"pair_style      deepmd {graph_list} out_freq {trj_freq} out_file model_devi.out\n "
                _dump_idx = check_keywords(lmp_lines, ['dump', 'dpgen_dump'])
                lmp_lines[_dump_idx] = \
                    f"dump            dpgen_dump all custom {trj_freq} dump.lammpstrj id type x y z fx fy fz\n"
                lmp_lines = substitute_keywords(lmp_lines, task)

                # revise input of plumed
                if use_plm:
                    _plm_idx = check_keywords(lmp_lines, ['fix', 'dpgen_plm'])
                    lmp_lines[_plm_idx] = \
                        "fix            dpgen_plm all plumed plumedfile input.plumed outfile output.plumed\n "
                    with open(plm_templ) as fp:
                        plm_lines = fp.readlines()
                    plm_lines = substitute_keywords(plm_lines, task)
                    with open(os.path.join(task_path, 'input.plumed'), 'w') as fp:
                        fp.write(''.join(plm_lines))

                # dump input of lammps
                with open(os.path.join(task_path, 'input.lammps'), 'w') as fp:
                    fp.write(''.join(lmp_lines))
                with open(os.path.join(task_path, 'job.json'), 'w') as fp:
                    job = task
                    json.dump(job, fp, indent=4)

                # dump init structure
                write(os.path.join(task_path, 'conf.lmp'),
                      sys_init_stc, format='lammps-data')

    def fp_error_test(
            self,
            machine_name,
            resource_dict,
            iteration=None,
            test_model=None
    ):
        """Test your model quickly with the data generated from tesla.

        Parameters
        ----------
        resource_dict : dict
        machine_name : str
        iteration : str, optional
            Select the iteration of data for testing. Default: the latest one.
        test_model : str, optional
            Select the iteration of model for testing. Default: the latest one.

        Return
        ------

        """

        location = self.task.path
        if iteration is None:
            if self.task.step_code < 7:
                iteration = self.task.iteration - 1
            else:
                iteration = self.task.iteration
        logger.info("Preparing structures from FP runs.")
        n_iter = 'iter.' + str(iteration).zfill(6)
        quick_test_dir = os.path.join(location, n_iter, '03.quick_test')
        os.makedirs(os.path.join(quick_test_dir, 'task.md'), exist_ok=True)
        task_list = glob(os.path.join(location, n_iter, '02.fp', 'task*'))
        task_list.sort()
        _stc_file = self._fp_output_style()
        _dpgen_output = self._fp_output_dpgen()
        _dpdata_format = self._fp_output_format()
        all_sys = None
        stcs = []
        for idx, oo in enumerate(task_list):
            logger.debug(f"Task: {oo}")
            sys = dpdata.LabeledSystem(os.path.join(
                oo, _dpgen_output), fmt=_dpdata_format)
            stc = read(os.path.join(oo, _stc_file))
            if len(sys) > 0:
                sys.check_type_map(type_map=self.task.param_data['type_map'])
            if idx == 0:
                all_sys = sys
                stcs.append(stc)
            else:
                try:
                    all_sys.append(sys)
                    stcs.append(stc)
                except (RuntimeError, TypeError, NameError):
                    pass
        write(os.path.join(quick_test_dir, 'task.md/validate.xyz'),
              stcs, format='extxyz')
        atom_numb = np.sum(all_sys['atom_numbs'])
        dft_energy = all_sys['energies']
        dft_force = all_sys['forces']
        if test_model is None:
            if self.task.step_code < 2:
                test_model = self.task.iteration - 1
            else:
                test_model = self.task.iteration
        model_iter = 'iter.' + str(test_model).zfill(6)
        model_dir = os.path.join(location, model_iter, '00.train')
        self._fp_generate_error_test(
            work_path=quick_test_dir, model_dir=model_dir)
        if not os.path.exists(os.path.join(quick_test_dir, 'task.md/conf.lmp')):
            _lmp_data = glob(os.path.join(location, n_iter,
                                          '01.model_devi', 'task*', 'conf.lmp'))[0]
            os.symlink(_lmp_data, os.path.join(
                quick_test_dir, 'task.md/conf.lmp'))
        logger.info("Quick tests task submitting.")
        job = self.md_single_task(
            work_path=quick_test_dir,
            model_path=model_dir,
            numb_models=self.task.param_data['numb_models'],
            forward_files=['conf.lmp', 'input.lammps', 'validate.xyz'],
            backward_files=['model_devi.out', 'energy.log',
                            'quick_test.log', 'quick_test.err', 'dump.lammpstrj'],
            outlog='quick_test.log',
            errlog='quick_test.err',
            machine_name=machine_name,
            resource_dict=resource_dict
        )
        job.run_submission()
        logger.info("Quick tests finished.")
        quick_test_result_dict = self._fp_error_test_result(
            quick_test_dir, atom_numb, dft_energy, dft_force)
        fig = self._fp_error_test_plot(iteration, **quick_test_result_dict)
        return quick_test_result_dict, fig

    @staticmethod
    def _fp_error_test_result(quick_test_dir, atom_numb, dft_energy, dft_force):
        md_energy = np.loadtxt(os.path.join(
            quick_test_dir, 'task.md/energy.log'), usecols=3)
        _md_stc = read(os.path.join(
            quick_test_dir, 'task.md/dump.lammpstrj'), index=':', format='lammps-dump-text')
        md_force = np.array([ss.get_forces() for ss in _md_stc])
        dft_energy_per_atom = dft_energy / atom_numb
        md_energy_per_atom = md_energy / atom_numb
        energy_per_atom_rmse = np.sqrt(
            np.mean((md_energy - dft_energy) ** 2)) / atom_numb
        md_force_r = np.ravel(md_force)
        dft_force_r = np.ravel(dft_force)
        force_rmse = np.sqrt(np.mean((md_force_r - dft_force_r) ** 2))
        result_dict = {
            "dft_energy_per_atom": dft_energy_per_atom,
            "md_energy_per_atom": md_energy_per_atom,
            "energy_per_atom_rmse": energy_per_atom_rmse,
            "dft_force": dft_force_r,
            "md_force": md_force_r,
            "force_rmse": force_rmse
        }
        return result_dict

    @staticmethod
    def _fp_error_test_plot(iteration, dft_energy_per_atom, md_energy_per_atom, energy_per_atom_rmse, dft_force,
                            md_force, force_rmse):
        from matplotlib.offsetbox import AnchoredText

        fig, axs = plt.subplots(1, 2)
        # Plot of energy error
        axs[0].scatter(dft_energy_per_atom, md_energy_per_atom,
                       s=5, label=f'Iter. {iteration}')
        _x = np.linspace(np.min(dft_energy_per_atom) - 0.05,
                         np.max(dft_energy_per_atom) + 0.05, 10)
        axs[0].plot(_x, _x, 'r--')

        box = AnchoredText(
            s=f'RMSE={energy_per_atom_rmse} (eV/atom)', prop=dict(fontsize=14),
            loc="upper left", frameon=False)
        box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        axs[0].add_artist(box)
        axs[0].set_title(f'Energy error', fontsize=14)
        axs[0].set_xlabel(r'$e_{DFT}$ (eV/atom)', fontsize=14)
        axs[0].set_ylabel(r'$e_{DPMD}$ (eV/atom)', fontsize=14)
        axs[0].set_aspect('equal')
        # Plot of force error
        axs[1].scatter(md_force, dft_force, s=5, label=f'Iter. {iteration}')
        _y = np.linspace(np.min(dft_force) - 0.05,
                         np.max(dft_force) + 0.05, 10)
        axs[1].plot(_y, _y, 'r--')
        axs[1].text(np.min(dft_force) - 0.05, np.max(dft_force) +
                    0.05, f'RMSE={force_rmse} (eV/Å)', fontsize=14)
        axs[1].set_title(f'Force error', fontsize=14)
        axs[1].set_xlabel(r'$f_{DFT}$ (eV/Å)', fontsize=14)
        axs[1].set_ylabel(r'$f_{DPMD}$ (eV/Å)', fontsize=14)
        axs[1].set_aspect('equal')
        return fig

    def _fp_generate_error_test(self, work_path, model_dir):
        model_list = glob(os.path.join(model_dir, 'graph*pb'))
        model_list.sort()
        model_names = [os.path.basename(i) for i in model_list]
        input_file = "units           metal\n"
        input_file += "boundary        p p p\n"
        input_file += "atom_style      atomic\n"
        input_file += "\n"
        input_file += "neighbor        2.0 bin\n"
        input_file += "neigh_modify    every 10 delay 0 check no\n"
        input_file += "read_data       conf.lmp\n"
        _masses = self.task.param_data['mass_map']
        # change the masses of atoms
        for ii, jj in enumerate(_masses):
            input_file += f"mass            {ii + 1} {jj}\n"
        # change the file name of graphs
        input_file += "pair_style deepmd "
        for kk in model_names:
            input_file += f"../{kk} "
        input_file += "out_freq 1 out_file model_devi.out\n"

        input_file += \
            """
            pair_coeff
            velocity        all create 330.0 23456789
            fix             1 all nvt temp 330.0 330.0 0.05
            timestep        0.0005
            thermo_style    custom step pe ke etotal temp press vol
            thermo          1
            dump            1 all custom 1 dump.lammpstrj id type x y z fx fy fz
            variable temp equal temp
            variable etotal equal etotal
            variable pe equal pe
            variable ke equal ke
            variable step equal step
            fix sys_info all print 1 "${step} ${temp} ${etotal} ${pe} ${ke}" title "#step temp etotal pe ke" file energy.log
            """
        input_file += "rerun validate.xyz dump x y z box no format xyz\n"
        with open(os.path.join(work_path, 'task.md/input.lammps'), 'w') as f:
            f.write(input_file)
        return input_file

    def _fp_style(self):
        styles = {
            "vasp": "POSCAR",
            "cp2k": "coord.xyz",
        }
        return styles.get(self.task.param_data['fp_style'], None)

    def _fp_output_style(self):
        styles = {
            "vasp": "vasprun.xml",
            "cp2k": "coord.xyz",
        }
        return styles.get(self.task.param_data['fp_style'], None)

    def _fp_output_dpgen(self):
        styles = {
            "vasp": "OUTCAR",
            "cp2k": "output",
            "qe": "output",
            "siesta": "output",
            "gaussian": "output",
            "pwmat": "REPORT",
        }
        return styles.get(self.task.param_data['fp_style'], None)

    def _fp_output_format(self):
        styles = {
            "vasp": "vasp/outcar",
            "cp2k": "cp2k/output",
        }
        return styles.get(self.task.param_data['fp_style'], None)
