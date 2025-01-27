import logging
import os
import json
from typing import List, Type, Dict, Any, Union
from .tools import InteractionTool, CacheInteractionTool
from .tools.evaluation_tool import RecommendationEvaluator, SimulationEvaluator
from .agent.simulation_agent import SimulationAgent
from .llm import LLMBase
from .agent.recommendation_agent import RecommendationAgent
from .tasks.simulation_task import SimulationTask
from .tasks.recommendation_task import RecommendationTask

logger = logging.getLogger("websocietysimulator")

class Simulator:
    def __init__(self, data_dir: str = None, block_set_dir: str = None, device: str = "auto", cache: bool = False):
        """
        Initialize the Simulator.
        Args:
            data_dir: Path to the directory containing Yelp dataset files.
            device: Device to use for evaluation. "auto" (default) will use GPU if available, otherwise CPU. Available options: "gpu", "cpu", "auto".
            cache: Whether to use cache for interaction tool.
        """
        logger.info("Start initializing Simulator")
        self.data_dir = data_dir
        self.block_set_dir = block_set_dir
        if data_dir is None:
            self.interaction_tool = None
        else:
            if cache:
                logger.info("Using CacheInteractionTool")
                self.interaction_tool = CacheInteractionTool(data_dir, block_set_dir)
            else:
                logger.info("Using Normal InteractionTool")
                self.interaction_tool = InteractionTool(data_dir, block_set_dir)
        
        self.tasks = []  # List to store tasks
        self.groundtruth_data = []  # List to store groundtruth data
        self.agent_class = None
        self.llm = None
        self.recommendation_evaluator = RecommendationEvaluator()
        self.simulation_evaluator = SimulationEvaluator(device)
        self.simulation_outputs = []
        self.evaluation_results = []
        logger.info("Simulator initialized")

    def set_interaction_tool(self, interaction_tool: Union[InteractionTool, CacheInteractionTool]):
        self.interaction_tool = interaction_tool

    def set_task_and_groundtruth(self, task_dir: str, groundtruth_dir: str):
        """
        Load tasks from a directory.
        Args:
            task_dir: Directory containing task files.
            groundtruth_dir: Directory containing groundtruth files.
        """
        self.tasks = []  # Clear previous tasks
        self.groundtruth_data = []

        # 获取所有task文件并按index排序
        task_files = sorted([f for f in os.listdir(task_dir) if f.startswith('task_') and f.endswith('.json')], 
                          key=lambda x: int(x.split('_')[1].split('.')[0]))

        for task_file in task_files:
            # 获取对应的groundtruth文件
            task_index = task_file.split('_')[1].split('.')[0]
            groundtruth_file = f'groundtruth_{task_index}.json'
            groundtruth_path = os.path.join(groundtruth_dir, groundtruth_file)
            
            if not os.path.exists(groundtruth_path):
                logger.warning(f"Groundtruth file {groundtruth_file} not found for task {task_file}")
                continue

            # 读取task文件
            task_path = os.path.join(task_dir, task_file)
            with open(task_path, 'r') as f:
                task_data = json.load(f)
                task_type = task_data.get('type')

                # Determine scenario type and create corresponding object
                if task_type == 'user_behavior_simulation':
                    task = SimulationTask(
                        user_id=task_data['user_id'],
                        item_id=task_data['item_id']
                    )
                elif task_type == 'recommendation':
                    task = RecommendationTask(
                        user_id=task_data['user_id'],
                        candidate_category=task_data['candidate_category'],
                        candidate_list=task_data['candidate_list'],
                        loc=task_data['loc']
                    )
                else:
                    raise ValueError(f"Unsupported task type: {task_type}")

            with open(groundtruth_path, 'r') as f:
                groundtruth_data = json.load(f)
                
            self.tasks.append(task)
            self.groundtruth_data.append(groundtruth_data)

        logger.info(f"Loaded {len(self.tasks)} task-groundtruth pairs")

    def set_agent(self, agent_class: Type):
        """
        Set the agent class to be used for the simulation.
        Args:
            agent_class: A class inheriting from the abstract Agent class.
        """
        if not issubclass(agent_class, (SimulationAgent, RecommendationAgent)):
            raise ValueError("Agent class must inherit from SimulationAgent or RecommendationAgent.")
        self.agent_class = agent_class
        logger.info("Agent class set")

    def set_llm(self, llm: Union[LLMBase, list[LLMBase]]):
        """
        Set the LLM to be used for the simulation.
        Args:
            llm: A class inheriting from the abstract LLM class.
        """
        self.llm = llm
        logger.info("LLM set")

    def run_simulation(self, number_of_tasks: int = None, enable_threading: bool = False, max_workers: int = None, time_limitation: float = None) -> List[Any]:
        """
        Run the simulation with optional multi-threading support and time limitation.
        
        Args:
            number_of_tasks: Number of tasks to run. If None, run all tasks.
            enable_threading: Whether to enable multi-threading. Default is False.
            max_workers: Maximum number of threads to use. If None, will use min(32, number_of_tasks).
            time_limitation: Time limit in minutes. If None, no time limit is applied.
        Returns:
            List of outputs from agents for each scenario.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

        start_time = time.time()
        timeout_seconds = time_limitation * 60 if time_limitation else None

        logger.info("Running simulation")
        if not self.agent_class:
            raise RuntimeError("Agent class is not set. Use set_agent() to set it.")
        if not self.interaction_tool:
            raise RuntimeError("Interaction tool is not set. Use set_interaction_tool() to set it.")

        task_to_run = self.tasks[:number_of_tasks] if number_of_tasks is not None else self.tasks
        logger.info(f"Total tasks: {len(task_to_run)}")

        # 如果不启用多线程，使用原始的串行处理
        if not enable_threading:
            self.simulation_outputs = []
            for index, task in enumerate(task_to_run):
                # 检查是否超时
                if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                    logger.warning(f"Time limit ({time_limitation} minutes) reached. Stopping simulation.")
                    break

                if isinstance(self.llm, list):
                    agent = self.agent_class(llm=self.llm[index%len(self.llm)])
                else:
                    agent = self.agent_class(llm=self.llm)
                agent.set_interaction_tool(self.interaction_tool)
                agent.insert_task(task)
                
                try:
                    output = agent.workflow()
                    result = {
                        "task": task.to_dict(),
                        "output": output
                    }
                except NotImplementedError:
                    result = {
                        "task": task.to_dict(),
                        "error": "Forward method not implemented by participant."
                    }
                self.simulation_outputs.append(result)
                logger.info(f"Simulation finished for task {index}")
        else:
            # 多线程处理
            from threading import Lock, Event
            
            log_lock = Lock()
            cancel_event = Event()  # 添加取消事件标志
            self.simulation_outputs = [None] * len(task_to_run)

            def process_task(task_index_tuple):
                from concurrent.futures import ThreadPoolExecutor, TimeoutError
                
                def run_agent_task(agent, task):
                    output = agent.workflow()
                    return output
                
                index, task = task_index_tuple
                # 检查是否已经被要求取消
                if cancel_event.is_set():
                    return index, None
                    
                if isinstance(self.llm, list):
                    agent = self.agent_class(llm=self.llm[index%len(self.llm)])
                else:
                    agent = self.agent_class(llm=self.llm)
                agent.set_interaction_tool(self.interaction_tool)
                agent.insert_task(task)
                
                try:
                    # 使用内部的ThreadPoolExecutor来执行单个任务，设置超时时间为5分钟
                    with ThreadPoolExecutor(max_workers=1) as single_task_executor:
                        future = single_task_executor.submit(run_agent_task, agent, task)
                        try:
                            output = future.result(timeout=300)  # 5 minutes timeout
                            result = {
                                "task": task.to_dict(),
                                "output": output
                            }
                        except TimeoutError:
                            logger.warning(f"Task {index} timed out")
                            # 强制关闭执行器
                            single_task_executor._threads.clear()
                            single_task_executor.shutdown(wait=False)
                            return index, None
                except NotImplementedError:
                    result = {
                        "task": task.to_dict(),
                        "error": "Forward method not implemented by participant."
                    }
                except Exception as e:
                    logger.error(f"Task {index} failed with error: {str(e)}")
                    return index, None
                
                with log_lock:
                    logger.info(f"Simulation finished for task {index}")
                
                return index, result

            # 确定线程数
            if max_workers is None:
                max_workers = min(32, len(task_to_run))
            else:
                max_workers = min(max_workers, len(task_to_run))
            
            logger.info(f"Running with {max_workers} threads")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(process_task, (i, task)): i 
                    for i, task in enumerate(task_to_run)
                }

                try:
                    for future in as_completed(future_to_index, timeout=timeout_seconds):
                        try:
                            index, result = future.result()
                            self.simulation_outputs[index] = result
                        except Exception as e:
                            logger.error(f"Task failed with error: {str(e)}")
                except TimeoutError:
                    logger.error(f"Time limit ({time_limitation} minutes) reached.")
                    # 设置取消标志
                    cancel_event.set()
                    # 强制取消所有任务
                    for future in future_to_index:
                        future.cancel()
                    # 立即关闭执行器，不等待任务完成
                    executor._threads.clear()
                    executor.shutdown(wait=False)
                    raise TimeoutError

        logger.info("Simulation finished")
        # 过滤掉None值（未完成的任务）
        return self.simulation_outputs

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the simulation results using the loaded groundtruth data.
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating simulation results")
        if not self.simulation_outputs:
            raise RuntimeError("No simulation outputs to evaluate. Run simulation first.")
        
        # 检查数据条目数量
        sim_count = len(self.simulation_outputs)
        gt_count = len(self.groundtruth_data)
        
        if sim_count != gt_count:
            logger.warning(f"Warning: Number of simulation outputs ({sim_count}) does not match ground truth data ({gt_count})")
            # 使用较小的数量
            eval_count = min(sim_count, gt_count)
            groundtruth_data = self.groundtruth_data[:eval_count]
            self.simulation_outputs = self.simulation_outputs[:eval_count]
        else:
            groundtruth_data = self.groundtruth_data
        
        evaluation_results = {}
        
        # 根据agent类型选择评估方法
        if issubclass(self.agent_class, RecommendationAgent):
            evaluation_results = self._evaluate_recommendation(groundtruth_data)
        elif issubclass(self.agent_class, SimulationAgent):
            evaluation_results = self._evaluate_simulation(groundtruth_data)
        
        # 添加数据条目信息到评估结果中
        evaluation_results['data_info'] = {
            'evaluated_count': eval_count if sim_count != gt_count else sim_count,
            'original_simulation_count': sim_count,
            'original_ground_truth_count': gt_count
        }
        
        self.evaluation_results.append(evaluation_results)
        logger.info("Evaluation finished")
        return evaluation_results

    def _evaluate_recommendation(self, ground_truth_data: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate recommendation results using groundtruth
        """
        # 从ground truth数据中提取真实POI
        gt_pois = [item['ground truth'] for item in ground_truth_data]
        
        pred_pois = []
        for output in self.simulation_outputs:
            if output is not None:
                pred_pois.append(output['output'])
            else:
                pred_pois.append([''])

        # 计算评估指标
        metrics = self.recommendation_evaluator.calculate_hr_at_n(
            ground_truth=gt_pois,
            predictions=pred_pois,
        )

        return {
            'type': 'recommendation',
            'metrics': metrics.__dict__,
        }

    def _evaluate_simulation(self, ground_truth_data: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate simulation results
        """
        simulated_data = []
        for output in self.simulation_outputs:
            if output is not None:
                simulated_data.append(output['output'])
            else:
                simulated_data.append({
                    'stars': 0,
                    'review': ''
                })
        metrics = self.simulation_evaluator.calculate_metrics(
            simulated_data=simulated_data,
            real_data=ground_truth_data
        )
        return {
            'type': 'simulation',
            'metrics': metrics.__dict__,
        }

    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of evaluation results
        Returns:
            List of evaluation results
        """
        return self.evaluation_results