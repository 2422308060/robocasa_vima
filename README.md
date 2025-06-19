# 使用文档
> 需额外克隆`wbvima`和`robocasa`仓库并安装环境。
1. `split_hdf5_dataset.py`

   robocasa的数据集的一个任务封装在一个.hdf5，运行这个脚本将其拆分为train和val子集。
   
   修改该文件中`source_file`路径为下载好的robocasa数据集路径并运行，拆分后文件将保存在上述路径：
   
    ```
    python split_hdf5_dataset.py
    ```

2. `robocasa_converter_brs.py`

   将拆分后的train和val文件分别转换给wbvima训练用的格式。
   
   根据需求修改该文件中的`target_filename`,`robocasa_base_dir`,`robocasa_dataset_name`,`output_hdf5_dir`，运行脚本后文件将保存在`output_hdf5_dir`

    ```
    python robocasa_converter_brs.py
    ```

> 使用训练好之后得到的模型权重。训练过程参考[训练](https://behavior-robot-suite.github.io/docs/sections/wbvima/train.html)。用server_robocasa.py和client_brs.py来查看在仿真环境中的效果（细节处理参考代码内容）。
3. `server_robocasa.py`

   作为服务端放在robocasa环境和robocasa目录下运行

    ```
    python server_robocasa.py
    ```

4. `client_brs.py`

   作为客户端放在brs环境和brs-algo目录下运行

   修改该文件中的`CKPT_PATH`和`CONFIG_PATH`为训练后得到的.pth和.yaml路径


    ```
    python client_brs.py
    ```

    运行成功后会弹出窗口展示机器人根据模型的指令开始移动
