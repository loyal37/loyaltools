# LoyalTools

明日方舟：终末地（Arknights: Endfield）Mod 制作 Blender 插件。

基于 **TheHerta4** 改造，集成了 **EFMI-Tools** 的帧分析数据解析能力，实现从 **提取模型 → 编辑 → 标记贴图 → 导出 Mod** 的完整流程，**无需安装 SSMT4**。

- 面板位置：Blender 3D 视图 → 侧边栏（N 键）→ `LoyalTools` 标签页
- Blender 版本要求：4.5+

## 与 TheHerta4 的区别

| 功能 | TheHerta4 | LoyalTools |
|---|---|---|
| 模型提取 | 依赖 SSMT4 外部程序 | 内置「提取模型 (DrawIB)」面板，填入 DrawIB 直接从帧分析 Dump 提取 |
| 贴图标记 | 在 SSMT4 中标记 | 内置「标记贴图」面板，在 Blender 内完成 |
| 游戏预设 | 从 SSMT4 配置读取 | 独立模式下可直接选择（默认 EFMI/终末地），也兼容已安装的 SSMT4 |
| 工作空间 | SSMT4 管理 | 自定义目录即可（SSMT4 存在时也可同步） |
| 蓝图导出 / 工具集 | ✔ | ✔（完整保留） |
| EFMI 骨骼合并 | 需外部流程 | 内置独立提取/导入流程，Cross IB 节点可切换一般跨 IB / 骨骼合并 |

## 使用前提

1. 游戏侧需要已部署 **EFMI 运行时框架**（即提供 `CommandList\EFMIv1\OverrideTextures` 的框架 ini，随 XXMI 的 EFMI Importer 或 SSMT4 部署）。普通 Mod 依赖该命令列表；骨骼合并 Mod 要求 **EFMI 1.4.1+**。
2. 帧分析 Dump 需由 3dmigoto 生成（游戏内按 F8），d3dx.ini 中建议的分析选项：

   ```ini
   analyse_options = deferred_ctx_immediate dump_rt dump_cb dump_vb dump_ib dump_tex buf txt dds symlink
   ```

   提取器需要 Dump 目录内包含 `log.txt`、`deduped/` 子目录、各 VB/IB 的 `.buf` + `.txt` 文件以及贴图 `.dds`。

## 使用流程

### 1. 设置工作空间（独立模式）

「基础信息」面板 → 工作空间来源选择 **自定义目录**，填入任意空文件夹路径。
未检测到 SSMT4 时面板会显示「独立模式」提示；游戏预设默认为 EFMI（终末地）。
若你安装了 SSMT4 但想强制使用 LoyalTools 的预设，勾选「强制使用独立预设」。

### 2. 提取模型

「提取模型 (DrawIB)」面板：

1. 填入 **帧分析 Dump 目录**（FrameAnalysis-xxx 文件夹）
2. 在 DrawIB 分行表格中添加一个或多个 **DrawIB**（8 位十六进制），别名可留空
3. 点 **提取**，插件会同时复制候选贴图、导入网格并自动接入蓝图
4. 提取结果写入工作空间的 `workplace/` 子文件夹（`workplace/<DrawIB>-<索引数>-<起始索引>/TYPE_GPU-EFMI/` 结构，与 SSMT4 兼容），默认自动导入到场景（模型名为 `DrawIB-索引数-起始索引`，集合以自定义文件夹命名并标红）

### 3. 标记贴图

「标记贴图」面板：

1. 选中一个已导入的模型对象，点 **刷新贴图列表**
2. 列表显示该 DrawIB 提取到的候选贴图（含 ps-t 插槽与 hash 信息）
3. 选择标记名称（DiffuseMap / NormalMap / LightMap / 自定义）与标记方式：
   - **Slot 插槽方式**：ini 中直接绑定到 ps-t 插槽（推荐）
   - **Hash 方式**：按贴图 hash 生成独立的 TextureOverride
4. 点 **标记所选贴图**。贴图会复制为 `<DrawIB>-<索引数>-<起始索引>-<标记名>.dds`（SSMT4 风格命名），标记写入该子网格的 SubmeshJson，导出时自动生成贴图 ini 并拷贝贴图文件
5. 可用 **预览** / **打开贴图文件夹** 辅助辨认贴图

### 3.5 骨骼合并制作（独立流程）

骨骼合并不会改变普通 DrawIB 制作流程。在「提取模型」面板把「制作流程」切换为 **骨骼合并**：

1. 选择角色完整显示时抓取的 FrameAnalysis 文件夹，不需要填写 DrawIB。
2. 点「提取」。LoyalTools 会按 EFMI-Tools v0.6.2 的规则识别主要角色、读取各组件骨骼矩阵并建立全局顶点组映射。角色可同时包含显式权重组件和隐式权重组件；隐式组件会按 EFMI 规则自动给每个顶点的首个骨骼补 1.0 权重。
3. 输出目录和模型名仍为 `<IBHash>-<IndexCount>-<FirstIndex>`；工作空间根目录会额外生成 `EFMI_MergedSkeleton.json`。当前版本暂不提取 LOD。
4. 在骨骼合并模式点「导入」时，各组件的本地顶点组会自动改为 profile 中的全局顶点组；每个物体都会建立同一套 `0..bones_count-1` 顶点组空间，空组用于保持 Blender 组索引与全局骨骼 ID 一致。普通模式导入不会应用这份映射。
5. 不使用跨 IB 时无需额外节点，自动生成的蓝图和物体标记会让导出器直接进入骨骼合并模式。需要跨 IB 映射时再加入 Cross IB 节点，把「跨 IB 方式」切换为 **骨骼合并**；映射仍填写「源 IndexCount >> 目标 IndexCount」，不需要 VS 200–204 选项。
6. 生成 Mod。该模式会导出 16 位 `BLENDINDICES`、EFMI 1.4.1 Merged Skeleton 回调和各组件 EntryPoint；不会生成一般跨 IB 使用的 HLSL。只有蓝图中实际存在网格物体的 IB 才会生成 EntryPoint、绘制回调和网格文件；从蓝图删除眉毛、睫毛等物体后，对应 IB 不会被 Mod 接管，游戏会继续使用原版网格。

同一张蓝图不能混用「一般跨 IB」和「骨骼合并」。全局顶点组 ID 上限为 65535；单个游戏组件原始骨骼仍受游戏骨骼缓冲区的 256 项范围约束。

### 4. 编辑与导出

与 TheHerta4 完全一致：使用蓝图系统（或「快速局部导出」）生成 Mod。
独立模式下若未配置游戏目录，Mod 默认输出到 `<工作空间>/GeneratedMod/`，将其中内容放入游戏的 Mods 目录即可。

## 注意事项

- **不要与 TheHerta4 同时启用**：两者共享大量内部标识符（操作符/面板 ID），同时启用会注册冲突。
- 提取功能为终末地（EFMI）专用；其它游戏预设下会弹出警告。
- CPU 蒙皮（cpu_posed）的部件仅支持贴图替换，与 EFMI-Tools 的限制一致。
- 更新器指向 https://github.com/loyal37/loyaltools，「检查版本更新」从该仓库的 Release 获取新版本。

## 致谢与来源

- **TheHerta4**（GPL-3.0）— 本插件的主体框架
- **EFMI-Tools**（作者 SpectrumQT 等）— `efmi_extract/` 目录下的帧分析解析与数据类型代码移植自该项目。EFMI-Tools 未附带开源许可证，如需公开分发本插件，请先取得原作者授权。
