# GPT-SoVITS Sillytavern驱动器

## 项目功能
主要目的是解决sillytavern无法使用最新版本的痛点（主要还是GPT-SoVITSv2pro和proplus的模型很多，而且效果相对v2还好不少）

## 项目亮点
为什么要写这个呢...

该项目采用中间件的思想，对sillytavern发送TTS的包进行截获修改，然后转发给后端GPT-SoVITS

方便快捷，与GPT-SoVITS本体解耦，可以随着GPT-SoVITS的版本更新而不需要重新配置。

提供多个hook节点，可以对tts发送的数据进行预处理，并且对GPT-SoVITS返回的数据进行后处理。

譬如自带的翻译插件，让GPT-SoVITS可以采用训练时的语言文本进行阅读，减小违和感，增加沉浸感。

项目基本上全注释了，想魔改或者自己改点东西的可以用hook，也可以自己看对应的位置魔改。


## 快速开始
视频教程：[占位符]
本项目需要Python3.8及以上环境运行（你都能跑GPT-SoVITS了，用他给的python环境即可）

首先配置config.py的内容，填入你自己的GPT-SoVITS后端地址。

然后修改当中的目录设置（可选，主要还是看你的参考音频打算放哪，为了避免某些系统不支持，尽量保持默认或者选择英文）

目录设置默认为voice，首次启动后，你可以在你的运行目录看到voice文件夹自动创建

向其放入对应的参考音频和参考文本即可。参考文本为同名文本文档。
譬如放入anno.wav,anno.txt

程序会自动将其解释为参考音频和参考文本。

GPT-SoVITS允许不添加参考文本，但是效果会差很多，建议这里还是给出参考文本。

然后在models.json中填入你的GPT和SoVITS文件路径即可。语言可填可不填，如果你是采用的动漫角色语音，一般是ja（日语），否则是zh（中文）

参考样例：

'''json
{
    # 这里填的是你的参考音频文件名
    "anno": {
        # 填入你的gpt和sovits路径，需要填入绝对路径。
        "gpt": "D:/GPT-SoVITS/GPT_weights_v2ProPlus/anno-e15.ckpt",
        "sovits": "D:/GPT-SoVITS/SoVITS_weights_v2ProPlus/anno_e8_s6016.pth",
        #可以不填，默认为中文，这里的默认可以在config中修改
        "prompt_lang": "ja"
    }
}
'''

你要真不会填建议问ai，把你的东西给他，让ai帮你写了再回来，json比较坑的就是容易忘逗号了。到时候考虑改成toml。

这样本地就配置好了。

默认翻译是关闭的，如果需要打开，则需要配置硅基流动的api key后才可正常使用。

只要不是大面积翻译，默认的14B已经很够用了。而且可以用很久很久。


接下来来到sillytavern，找到拓展页面（如果你没装稀奇古怪的各类插件，是从右往左的第三个，笑脸左边那个方块跺）
找到TTS，点击进去

服务提供商选择GPT-SoVITS-V2(Unofficial)

勾选已启用

在下边找到Provider Endpoint：

填入程序监听的地址，如果你是自己搭在本地的，并且没有修改端口（如果你不知道什么是端口，那么就没修改）

填入

http://127.0.0.1:9881

即可。

点击下边的Available Voice，获取角色可用语音数据（其实就是看你的models.json和voice文件夹）

在上边选择卡片的说话人即可配置成功。

然后就可以听到ai的话啦。


## 插件配置
其实插件很好配置，填入硅基流动的api key就差不多了

~~但考虑到amily2都有人不会用~~


## TODO LIST
[] 添加更多hook节点

[] 兼容GPT-SoVITSv3 和v4

[] 添加更多插件
s
[] 彻底优化sillytavern官方API

[] 配置改为toml亲民化




