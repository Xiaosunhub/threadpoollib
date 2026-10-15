# 项目长期记忆：threadpoollib

## 概况
- C++17 静态库线程池（header-only 模板 + 单个 translation unit 实现）。
- 通过 CMake 构建并安装，`ports/` 下是 vcpkg port 配置，`dist/` 是上次 install 的产物副本（gitignored）。

## 代码风格（务必沿用）
- Allman 花括号，缩进 4 空格。
- 控制关键字后**不加空格**：`if(x)`、`for(...)`、`while(...)`；圆括号内两侧**加空格**：`f( a, b )` 式排版常见于跨行。
- 大量解释性注释说明「为什么」，注释与代码间通常空两行。
- 锁相关成员函数名以 `_locked` 结尾表示「调用者必须已持有 mtx_」。
- 私密辅助行为前缀常见于 `maybe_ / enqueue_ / discard_ / reap_`。

## 构建环境（Windows / 本机）
- 没有 g++ / clang++；CMake 找不到 VS 实例（`Visual Studio 17 2022` generator 失败）。
- 可用方式：`cl.exe` = `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe`。
  必须先 `call "...\VC\Auxiliary\Build\vcvars64.bat"`；在 Git Bash 里直接转义调用不可靠，写临时 .bat 脚本再用 `cmd //c` 执行。
- 注意 bash 的 `/tmp` 映射到 `C:\Users\thinkpad\AppData\Local\Temp`，而 Write 工具写的 `C:/tmp/...` 是真实盘符路径，两者不是同一个目录。
