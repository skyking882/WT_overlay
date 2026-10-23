# 技术文档：直接导入 .wrpl 回放（fork 解析器、预编译后端、沙盘拖入和命令行）

（2026-10-23。用户："我希望用户直接把 wrpl 传入就好了"；选定：沙盘网页拖入 + 命令行；Windows 游戏电脑和 Mac 都要；
不改写成 Python；解析器源码也上传 GitHub，用 fork 上游 + 子模块；公开给 GitHub 用户。本文是规格，尚未实现。
标 D 的是我替用户定的默认值或未验证的假设，实现时要核实。）

## 背景

- **现在的导入要五步：**
  1. 切到 missle_sim 本地分支 `wt-overlay-exports`；
  2. 克隆 LivingTheDagor/WrplReplayParser@221be5ca，跑 11 个补丁脚本，用 CMake 编译；
  3. 准备回放同版本的 `aces` / `game` / `char.vromfs.bin`；
  4. 用 `inspect_match.py` 导出 `match_export.json`；
  5. 用 `scripts/wt_replay_import.py` 转成 `outputs/engagements/wt_real/<名字>.jsonl`。

  Mac 上还要用 x86 版的 Python 在 Rosetta 下运行，Windows 上要走 WSL。
- **上游本身支持 Windows 原生编译：** README 写明支持 MSVC 和 MSYS2 MinGW，CMake 里也有 MSVC 分支。WSL 只是我们构建脚本的选择。
- **上游 Python 绑定：** pybind11，CMake 缺省找 Python 3.12，pybind11 在配置时用 FetchContent 拉 v3.0.4。
- **后端读的游戏文件：** `replay/init/init.cpp` 从游戏目录挂载以下文件：
  - 必需：`aces.vromfs.bin`、`game.vromfs.bin`、`char.vromfs.bin`；
  - 可选：`mis`、`lang`、`ui/fonts`，`inspect_match.py` 都关掉了；
  - 可选：`content/pkg_local`，即 `--local-package`。
- **macOS：**
  - 现在的构建是 x86_64，因为 Dagor 的向量数学缺省走 SSE4；
  - Dagor 的 `dag_vecMathDecl.h` 在 `__ARM_NEON` 下会改走 `dag_vecMath_neon.h`，所以 arm64 原生**可能**编得过（D，没试过）。
- **许可证（2026-10-23 逐目录查过，固定版本 221be5c）：**
  - 上游：BSD 3-Clause；
  - `3rdParty/Dagor_Engine`：BSD 3-Clause。文件头的 "Copyright (C) Gaijin Games KFT. All rights reserved." 是 Dagor 开源版的标准文件头，许可仍是 BSD；
  - EASTL、zstd（`arc/zstd-1.4.5`）、tracy：BSD；
  - fmt、mimalloc、cpptrace、libdeflate、libunwind、enet、hash：MIT 类；
  - compile-time-regular-expressions：Apache-2.0 with LLVM exception；
  - boost（describe / mp11）：BSL-1.0；
  - zlib-ng、tinyfiledialogs：zlib；
  - imgui、tracy 目录里只有 CMake，构建时从各自仓库拉源码；
  - **例外是 `3rdParty/regExp`：** Digital Mars，"All Rights Reserved"，目录里放的是空文件 `NO_LICENSE`。它没有参与构建（`3rdParty/CMakeLists.txt` 里 `# add_subdirectory(regExp)`），**fork 里删掉**。

## 1. 源码放在哪里

### 1.1 Fork（`skyking882/WrplReplayParser`，公开）

- 用户在 GitHub 网页上 fork `LivingTheDagor/WrplReplayParser`。这台 Mac 没装 `gh`，所以这一步由用户做。
- 新分支 `wt-overlay`，从 `221be5cac8e772ab41eefb0e7dac35ce6fb5c39b` 开出。
- **每个补丁脚本做成一个提交：** 提交说明取补丁脚本的文档字符串，按 `build_backend_macos.sh` 里的顺序：
  1. fix_linux_includes
  2. fix_callback_shutdown
  3. fix_local_package
  4. fix_blk_include
  5. fix_custom_weapon_slots
  6. enable_weapon_probe
  7. preserve_weapon_raw
  8. fix_ecs_scalar_bindings
  9. retain_fm_sync
  10. retain_unit_mpi
  11. expose_kill_fields
  12. fix_macos_build
- **验收：** 分支最后的源码树和"固定版本 + 依次跑完 12 个脚本"得到的树逐文件相同。`git diff --stat` 为空，例外只有下面两项。
- **另加两类提交：**
  - 删除 `3rdParty/regExp`；
  - 新增 `MODIFICATIONS.md`，列出我们改了什么、为什么改，并注明上游版权和许可不变。`LICENSE` 和各第三方目录的许可证文件原样保留。
- 以后为 MSVC 或 arm64 做的修改也作为单独提交加在这个分支上，**不改解析逻辑**；要改解析逻辑必须另说明理由。
- 提交作者用与 WT_overlay 相同的 git 身份。推送前做隐私扫描，命中数必须为 0。
- missle_sim 里的补丁脚本保持原样，留作研究工具，WT_overlay 不再用它们。

### 1.2 WT_overlay

- **子模块：** `third_party/WrplReplayParser` 指向 fork，固定到 `wt-overlay` 分支的某个提交，URL 用 https。
- **新包 `wt_overlay/wrpl/`：**

  | 文件 | 作用 |
  |---|---|
  | `export.py` | 由 missle_sim 的 `inspect_match.py`（275 行）移植过来。**只在后端可执行文件里运行**，是唯一 import `PyReplayParser` 的代码。输出与原脚本逐字段相同 |
  | `backend.py` | 找后端可执行文件：本地路径、缓存、下载；下载后校验 SHA-256 |
  | `gamefiles.py` | 找游戏文件目录（§4） |
  | `importer.py` | 一次导入：.wrpl → 后端子进程 → `match_export.json` → 转换 → `wt_real/*.jsonl` |
  | `__main__.py` | 命令行（§6） |
  | `backend.json` | 后端版本、各平台的 Release 下载地址、SHA-256、大小 |

- **转换：** 复用 `scripts/wt_replay_import.py` 的逻辑，用 importlib 加载或把核心移进包里都可以。要求是 `scripts/wt_replay_import.py` 的命令行和输出**逐字节不变**，现有测试（`tests/test_sandbox_replay.py`、`tests/test_wt_replay_import_timing.py`）照样通过。
- **许可声明：** `THIRD_PARTY_NOTICES.md` 增加：
  - fork 的说明；
  - 后端可执行文件里打包的全部组件及其许可证原文：上表所有库、pybind11（BSD）、CPython（PSF）、PyInstaller 引导程序（GPL-2.0 + bootloader exception，允许打包分发）。

## 2. 后端可执行文件 `wrpl_export`

- **内容：** PyReplayParser 模块、Python 运行时和 `export.py`，用 PyInstaller 的 one-folder 模式冻结后打成 zip。用户不需要装 Python 版本匹配的解释器，也不需要编译器或 WSL。
- **命令行约定：**
  ```
  wrpl_export --game DIR --replay FILE --output DIR [--local-package DIR]
  wrpl_export --version
  ```
  - 成功：退出码 0，`DIR/match_export.json` 和 `DIR/parser.log`；
  - 失败：非 0，stderr 最后几行是原因，加上 `parser.log` 的末尾；
  - `--version` 打印后端版本、上游版本号和 fork 提交号（JSON 一行）；
  - 输出目录必须事先不存在，以免覆盖。
- **平台：**

  | 资产 | 构建 |
  |---|---|
  | `wrpl_export-windows-x64.zip` | MSVC（windows-latest），Python 3.12 |
  | `wrpl_export-macos-arm64.zip` | 先试原生 arm64（Dagor NEON）。成功且 §8.4 的对照一致才发布 |
  | `wrpl_export-macos-x64.zip` | x86_64（Intel runner，或在 arm64 上交叉编译加 universal2 Python，后者是本地已跑通的路径）。Apple Silicon 在 Rosetta 下运行 |
  | `wrpl_export-linux-x64.zip` | GCC（ubuntu-latest），给 CI 冒烟测试和 Linux 用户 |

- **macOS 选择顺序：** arm64 版存在就用 arm64，否则用 x64 版。x64 版在 Apple Silicon 上需要 Rosetta，没装时报中文错误，提示安装命令 `softwareupdate --install-rosetta`。
- 冻结后的可执行文件不签名（D）。macOS 解压后可能要去掉隔离属性，由下载代码自己去掉；用户手动下载时，README 写明怎么处理。

## 3. 构建、发布和下载

- **工作流：** `.github/workflows/wrpl_backend.yml`，在 WT_overlay 公开仓库。
  - 触发：手动触发，或推送 `wrpl-backend-v*` 标签；
  - 步骤：带子模块检出 → 编译 → PyInstaller → `--version` 冒烟测试 → 打 zip、生成 `SHA256SUMS` → 上传到同名 Release；
  - Release 附件里同时放 `THIRD_PARTY_NOTICES.md`；
  - CI 里**没有游戏文件也没有回放**，二者都不能随仓库分发，所以 CI 只能证明编得出、跑得起，解析结果的对照在本地做（§8.4）。
- **`backend.json`：** 发布后把各平台的 URL、SHA-256 和大小写进去，随代码提交。
- **下载：**
  - 第一次导入时，如果缓存里没有对应版本的后端，沙盘显示"需要下载解析后端（约 N MB，来自本项目 GitHub Release）"，用户点按钮才下载。命令行在交互终端里问 y/N，非交互时需要 `--download-backend`；
  - 下载完校验 SHA-256，不符就删掉并报错；
  - 解压到用户缓存：Windows `%LOCALAPPDATA%\wt_overlay\wrpl\<版本>`，macOS `~/Library/Caches/wt_overlay/wrpl/<版本>`，Linux `~/.cache/wt_overlay/wrpl/<版本>`；
  - 环境变量 `WT_WRPL_BACKEND` 或参数 `--backend PATH` 指向本地自编的后端时，跳过下载。
- **网络请求只有这一个下载，** 不上传任何东西，也没有统计。

## 4. 游戏文件

- **需要的文件：** 同一目录下的 `aces.vromfs.bin`、`game.vromfs.bin`、`char.vromfs.bin`，版本必须和回放一致；`content/pkg_local` 存在时自动挂载。只读，不改游戏文件。
- **查找顺序：**
  1. 命令行 `--game` 或沙盘里填的路径；
  2. 记住的上次路径，存在 `outputs/wrpl_settings.json`，`outputs/` 已被 git 忽略（D）；
  3. 回放所在目录的上一级：游戏把回放存在 `<游戏目录>/Replays/` 下（D，在游戏电脑上核实）；
  4. Windows：Steam 各库（解析 `libraryfolders.vdf`）下的 `steamapps/common/War Thunder`，以及 Gaijin 启动器的常见安装位置（D，路径待在游戏电脑上核实）；
  5. 都找不到时，要求用户指定。

  Mac 上通常没有游戏安装，可以指定一个放了这三个文件的文件夹，比如从游戏电脑拷过来的。
- **版本不一致：** 回放和游戏文件版本不同时，后端可能失败，也可能解出错误数据。
  - 实现时查清两边的版本号各从哪里读：回放头、vromfs 头，或游戏目录里的版本文件；
  - 能读到就在调用后端前比较，不一致时报"回放版本 X，游戏文件版本 Y"；
  - 读不到就在后端失败时附上这个提示；
  - 回放的版本号写进输出头的 `source`。

## 5. 沙盘

- **入口：**
  - WT 回放列表上方加"导入 .wrpl"按钮（选文件）；
  - 整个回放列表区域接受拖入，可以多个文件，排队逐个导入。
- **接口：** `POST /api/wrpl?name=<原文件名>`。
  - 请求体 `application/octet-stream`，上限 1 GiB（D）；
  - 本机地址、Origin、`X-Sandbox-Token` 检查和 `/api/command` 一样；
  - 边收边写到 `outputs/wrpl_inbox/` 下的临时文件，不整份读进内存；
  - 返回任务 id。
- **任务：**
  - 同一时间只跑一个导入，其余排队；
  - 阶段：保存 → 解析（后端）→ 转换 → 完成，通过现有 `/api/state` 的任务状态显示进度；
  - 后端子进程设超时，D：20 分钟；
  - 成功后刷新回放列表并选中新回放；
  - 失败时显示中文原因和后端日志末尾几行。
- **命名：**
  - 输出文件名取 .wrpl 文件名，去掉非法字符，重名时加后缀；
  - 按 SHA-256 记录导入过的回放，同一文件再次导入时直接定位到已有的回放，除非用户选"重新导入"。
- **找不到游戏文件时：** 弹出路径输入框。浏览器拿不到本地文件夹的真实路径，所以只能手填，服务器端校验三个文件是否存在。
- **后端未下载时：** 显示 §3 的下载确认按钮。

## 6. 命令行

```
python -m wt_overlay.wrpl 回放.wrpl [更多.wrpl 或文件夹 ...] [--game DIR] [--out-dir DIR] [--name NAME]
                          [--backend PATH] [--download-backend] [--keep-export] [--force]
```

- 文件夹参数导入其中所有 `.wrpl`，不递归（D）。
- 已导入过的（按 SHA-256）跳过，除非 `--force`。
- `--keep-export` 保留中间的 `match_export.json`，缺省删掉。
- 其余转换参数（`--team`、`--all-units` 等）照 `scripts/wt_replay_import.py` 透传。
- 启动脚本 `start_sandbox.cmd` / `.command` 不变。

## 7. 隐私

- 导入的回放、中间导出和输出都在本机 `outputs/` 下，被 git 忽略。
- 本机输出里保留玩家名，因为这是用户自己的数据。发布规则不变：回放不上 GitHub，要公开的示例必须先把所有玩家名换成队伍代码。
- 除 §3 的后端下载外没有任何网络请求。

## 8. 测试

1. **假后端：** 一个 Python 脚本，按 §2 的约定读参数，写出一份小的固定 `match_export.json`。用它端到端测试命令行和 `/api/wrpl`：
   - 正常导入；
   - 重复导入；
   - 后端失败、超时；
   - 缺游戏文件；
   - 超过大小上限；
   - 令牌或 Origin 错误。
2. **下载：**
   - 用本地 `file://` 地址，测试中不联网；
   - SHA-256 不符时拒绝并删除；
   - 缓存命中时不再下载；
   - `WT_WRPL_BACKEND` 优先于缓存。
3. **游戏目录查找：** 用假目录树测试，包括 `libraryfolders.vdf` 解析、回放上一级目录、记住的路径和优先顺序。
4. **真后端对照**（本地，按需运行，后端或游戏文件不存在时跳过）：用 2026-10-08 那批回放中的一个，比较冻结后端输出的 `match_export.json` 和现在已验证的 Mac x86 后端的输出。除路径和时间戳外逐字段相同，转换后的 `.jsonl` 逐字节相同。
   - Windows 版的这项对照由用户在游戏电脑上跑一条命令完成，我这边没有 Windows 机器；
   - arm64 版过了这项对照才发布。
5. **现有测试全部通过；** `scripts/wt_replay_import.py` 对已有导出的输出逐字节不变。

## 9. 实施顺序

1. **Fork 和补丁提交（§1.1）：** 先在本地完成并验收（源码树对比 + 本地 macOS 构建）。用户 fork 好以后，扫描结果为 0 且用户同意，才推送。
2. **WT_overlay 代码和测试（§1.2、§4–§8）：** 用假后端完成，全在本地。
3. **CI 工作流和首个 Release（§2、§3）：** 走 `.publish/publish.sh` 发布，用户同意后才推送和打标签。之后填 `backend.json`。
4. **真后端对照：** Mac 由我跑，Windows 由用户在游戏电脑上跑（§8.4）。
5. **README：** 加"导入 War Thunder 回放"一节。README 三个分支共用，`main`、`RL`、`cca-offense` 同时改。

## 10. 待定

- 功能进哪些分支：沙盘在 `main` 和 `RL` 上都有，D：两个都进，同地形那次的做法。
- §4 里 Windows 的安装和回放位置待在游戏电脑上核实。
- 版本号从哪里读（§4）。
- arm64 原生能否编过、结果是否一致（§2）。
- 冻结后端的体积；上传大小上限和超时（§5）。
- macOS 不签名带来的隔离属性处理方式（§2）。

## 11. 实现时的决定（2026-10-23）

- fork `wt-overlay` 分支在 12 个补丁之外还有三处构建修改，都不改解析逻辑：
  - 删除 `3rdParty/regExp`；
  - arm64 改用 NEON 并关闭浮点乘加融合，arm64 导出因此与 x86_64 逐字段相同；
  - cpptrace 不再用 libdwarf（LGPL-2.1），Unix 改用 libdl，Windows 用 dbghelp，只影响崩溃时的调用栈。
- 后端里的组件都是宽松许可，例外是 PyInstaller 引导程序：它是 GPL，但带允许分发的例外条款。
- 版本检查：回放和游戏文件只差构建号时放行，并在结果里注明；更大的差别报错，可用 `--ignore-version` 强行解析。
- 判断是否已导入：用 `.jsonl` 头里已有的 `source.replay_sha256`，不另建记录。
- 游戏文件查找：在"回放的上一级目录"之后，还查回放所在的目录本身。
- 转换器：一架飞机的头部导弹不再依赖哈希顺序，并列时 `X` 胜过 `X_default`，其余情况取最先发射的。
- CI：`wrpl-backend-v*` 标签建的是预发布版，不建草稿，因为草稿的附件不登录下载不了。本地对照（§8.4）通过后，才把地址和哈希写进 `backend.json`；在此之前不会下载任何后端。
- 运行沙盘不需要解析器源码，启动脚本不更新子模块。
