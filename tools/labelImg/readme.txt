基础开源代码：https://github.com/tzutalin/labelImg
在此基础上增加了pose, truncated标注

用法：
1. 环境安装（我是win10 64, 其他环境未测试）
  ● 安装Anaconda（目的是建立labelimg所需环境：python, pyqt, lxml等）
		下载地址： https://www.anaconda.com/download/
		我选择的是Anaconda5.0.1 python3.6 64-Bit, 有点大(515MB)
		双击安装，默认目录即可
  ● 点击桌面左下角cortana图标，搜寻anaconda, 会显示 Anaconda Prompt
  ● 运行 Anaconda Prompt
			> conda list						会看到所需的pyqt5, lxml已在其中
  ● 安装运行labelimg
		> cd labelimg
		> pyrcc4 -o resources.py resources.qrc
		> python labelImg.py
		demo实例：Open Dir选择 labelimg/demo/JPEGImages, Save Dir选择labelimg/demo/Annotations
2. 标注方法：
	用一哈就知道了。。。