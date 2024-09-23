from flask import Flask, jsonify
import datetime
import math
from _inspect_cuda import get_gpus

app = Flask(__name__)

# 用于处理 NaN 的辅助函数
def replace_nan(value):
    return None if math.isnan(value) else value

@app.route('/status', methods=['GET'])
def gpu_status():
    try:
        # 获取当前日期时间
        dt_str = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")

        # 获取 GPU 信息
        gpu_objects_dict, err_infos = get_gpus()
        gpu_info_list = list(gpu_objects_dict.values())

        if not gpu_info_list:
            raise ValueError("未能获取 GPU 信息")

        # 获取驱动和 CUDA 版本信息
        driver_versions = set([g.driver_version for g in gpu_info_list])
        cuda_versions = set([g.cuda_version for g in gpu_info_list])
        driver_version = (
            driver_versions.pop() if len(driver_versions) == 1 else tuple(driver_versions)
        )
        cuda_version = (
            cuda_versions.pop() if len(cuda_versions) == 1 else tuple(cuda_versions)
        )

        # 获取进程信息
        proc_info_list = [p for g in gpu_info_list for p in g.processes.values()]
        for i in range(len(proc_info_list)):
            # 设置全局索引
            proc_info_list[i].global_index = i

        # 组装返回数据
        return_data = {
            "datetime_str": dt_str,  # 当前时间
            "driver_version": driver_version,  # 驱动版本
            "cuda_version": cuda_version,  # CUDA 版本
            "gpu_info_list": [{
                "index": gpu.index,
                "name": gpu.name,
                "fan_speed": replace_nan(gpu.fan_speed),  # 替换 NaN
                "temperature_gpu": gpu.temperature_gpu,
                "power_draw": gpu.power_draw,
                "power_limit": gpu.power_limit,
                "memory_used": gpu.memory_used,
                "memory_total": gpu.memory_total,
                "utilization_gpu": gpu.utilization_gpu,
                "pcie_width_current": gpu.pcie_width_current,
                "pcie_gen_current": gpu.pcie_gen_current
            } for gpu in gpu_info_list],  # GPU 详细信息
            "proc_info_list": [{
                "global_index": proc.global_index,
                "gpu_index": proc.gpu_index,
                "container_name": proc.container_name,
                "proc_start_time": proc.proc_start_time,
                "proc_running_time": proc.proc_running_time,
                "pid": proc.pid,
                "pid_in_container": proc.pid_in_container,
                "process_name": proc.process_name,
                "gpu_memory_used": proc.gpu_memory_used,
                "main_memory_used": proc.main_memory_used,
                "command": proc.command
            } for proc in proc_info_list],  # 进程详细信息
            "err_infos": err_infos  # 错误信息
        }

        # 返回 JSON 响应
        return jsonify(return_data)

    except Exception as e:
        # 捕获所有异常，返回错误信息
        error_message = {
            "error": "获取 GPU 状态时发生错误",
            "message": str(e),
            "datetime_str": datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
        }
        return jsonify(error_message), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81, debug=True)

