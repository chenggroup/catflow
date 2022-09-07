from curses import use_default_colors
import json
import os
import sys
import shutil
from glob import glob
from pathlib import Path
from turtle import st

from yaml import load, SafeLoader
import numpy as np

from miko.utils import logger
from miko_tasker.utils.file import count_lines

from ase.io import read, write
from dpgen.remote.decide_machine import convert_mdata
from dpgen.generator.run import \
    make_train, run_train, post_train, \
    make_model_devi, run_model_devi, post_model_devi, \
    make_fp, run_fp, post_fp


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


# TODO: refine each step of whole workflow.
class DPTrain(TeslaWorkStep):
    def make(self):
        return make_train(self.step_code, self.params, self.machine)

    def run(self):
        return run_train(self.step_code, self.params, self.machine)

    def post(self):
        return post_train(self.step_code, self.params, self.machine)


class DPExploration(TeslaWorkStep):
    def make(self):
        return make_model_devi(self.step_code, self.params, self.machine)

    def run(self):
        return run_model_devi(self.step_code, self.params, self.machine)

    def post(self):
        return post_model_devi(self.step_code, self.params, self.machine)


class FPCalculation(TeslaWorkStep):
    def make(self):
        return make_fp(self.step_code, self.params, self.machine)

    def run(self):
        return run_fp(self.step_code, self.params, self.machine)

    def post(self):
        return post_fp(self.step_code, self.params)


class CLWorkFlow(object):
    def __init__(self, param_file, machine_pool):
        """initialize the concurrent learning workflow

        Parameters
        ----------
        param_file : a file containing all the parameters for workflow runs
        machine_pool : a file with machine for each step to use
        """
        self.param_file = Path(param_file).resolve()
        self.machine_pool = Path(machine_pool).resolve()
        self.stage = 0
        self.step = 0
        self.params = self.get_data(self.param_file)

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
        return convert_mdata(self.get_data(self.machine_pool))

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

    def check_converge(self):
        if self.main_step == 1:
            if self.real_step == 0:
                model_devi_jobs = self.params['model_devi_jobs']
                if self.stage >= len(model_devi_jobs):
                    return False
            else:
                return True
        else:
            return True

    def run_loop(self, record="miko.record"):
        record_file_path = Path(record).resolve()
        self.read_record(record_file_path)
        while self.check_converge():
            self.run_step()
            self.record_stage(record_file_path, self.stage, self.step)

    def render_params(self, param_file=None):
        params = self.params
        if param_file == None:
            param_file = self.param_file
            shutil.copyfile(
                self.param_file,
                self.param_file.parent / f"params_backup_{self.stage}.json"
            )
        with open(param_file, "w", encoding='utf-8') as f:
            json.dump(params, f, indent=4)


class ClusterReactionWorkflow(CLWorkFlow):
    def __init__(self, param_file, machine_pool, conf_file):
        super().__init__(param_file, machine_pool)
        workflow_settings = self.set_workflow(conf_file)
        self.workflow_settings = workflow_settings
        for key in workflow_settings.keys():
            setattr(self, key, workflow_settings[key])

    def set_workflow(self, conf_file):
        with open(conf_file) as f:
            conf = load(f, Loader=SafeLoader)
        return conf

    def check_converge(self):
        # check before model_devi step start
        if self.main_step == 1:
            if self.real_step == 0:
                model_devi_jobs = self.params.get('model_devi_jobs')
                try:
                    last_model_devi_job = model_devi_jobs[-1]
                except IndexError:
                    last_model_devi_job = {}
                last_sys_idx = last_model_devi_job.get('sys_idx')
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
                        finished_exploration = last_model_devi_job["_finished_exploration"]
                        if finished_exploration:
                            # all accu, finish exploration
                            logger.info('Model accuracy converged.')
                            if self.params.get("auto_long_train", False) == True:
                                logger.info('Long train task starts.')
                                long_train_task = LongTrain(self)
                                long_train_task.run_long_train()
                            return False
                        else:
                            logger.info("Exploration not ending, try again.")
                            self.update_params()
                            return True
            else:
                return True
        else:
            return True

    @staticmethod
    def _trust_limitation_check(sys_idx, lim):
        if isinstance(lim, list):
            sys_lim = lim[sys_idx]
        elif isinstance(lim, dict):
            sys_lim = lim[str(sys_idx)]
        else:
            sys_lim = lim
        return sys_lim

    def _get_trust_level(self, param_type, sys_idx):
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

    def update_params(self):
        updater = ClusterReactionUpdater(self)
        updater.model_devi_job_generator()
        self._set_cur_trust_level()
        self.render_params()

    def _set_cur_trust_level(self):
        f_trust_lo, f_trust_hi = self.guess_trust_level()
        cur_job = self.params['model_devi_jobs'][self.stage]
        cur_job["model_devi_f_trust_lo"] = f_trust_lo
        cur_job["model_devi_f_trust_hi"] = f_trust_hi

    def guess_trust_level(self, stage=None):
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
            if cur_job["sys_rev_mat"] == {}:
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
                cur_job["_finished_exploration"] = False
            else:
                for key in cur_job['sys_rev_mat'].keys():
                    cur_job['sys_rev_mat'][key]['lmp']['V_DIS2'] = cur_job['sys_rev_mat'][key]['lmp']['V_DIS1']

                task_list, distances = self._distances()

                is_coords = [i['lmp']['V_DIS1'][0]
                             for i in cur_job['sys_rev_mat'].values() if i.get('_type') == 'IS']
                logger.info(f"explored IS coords: {is_coords}")
                
                finished_exploration = True
                for new_coord in [
                    round(max(is_coords) + exploration_step, 3),
                    round(max(is_coords) + 2 * exploration_step, 3)
                ]:
                    if new_coord < self.exploration_track[center_idx]:
                        finished_exploration = False
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
                        finished_exploration = False
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
                cur_job["_finished_exploration"] = finished_exploration
        
        #TODO: generator for TS

        if add_new_flag == 0:
            cur_job["rev_mat"]["lmp"]["V_NSTEPS"] = [i * 2 for i in cur_job["rev_mat"]["lmp"]["V_NSTEPS"]]

    def _distances(self):
        task_list = []
        for i in range(self.workflow.workflow_settings['start_from_iter'], self.workflow.stage):
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
        self.workflow.params["sys_batch_size"].append("auto")
        return len(self.workflow.params["sys_configs"]) - 1

    def _get_cv_setting(self, cv_type):
        sys_idx = self.workflow.workflow_settings[cv_type]['sys_idx']
        coord = self.workflow.workflow_settings[cv_type]['coordination']
        return sys_idx, coord

class LongTrain:
    """Final step for CLWorkFlow"""
    def __init__(self, wf: CLWorkFlow) -> None:
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