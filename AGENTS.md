Chinese character involved in this project, mind encoding!
使用myml python环境,位于C:\Users\zyx20\anaconda3\envs\myml
我需要你帮助我开发强化学习程序,模拟滑翔机在风场中飞行.
mistakes部分中记录了过去常犯的错误，必须阅读保证不再犯错。如果有新的frequent mistake同样记录在mistakes部分中

在开始较复杂的非只读任务前，向用户询问详细的技术细节。完成和用户的alignment后在docs文件夹中写入/修改对应模块的.md定义文件，保持定义简洁。同时也检查其他文档是否符合代码状态和用户意图。

开发规则：
1. 如无必要禁止修改代码原有结构。修改结构需给出清晰依据

2. DRY,拒绝知识的重复。系统中的配置应有且仅有一个权威定义。禁止在多个地方手动同步相同的逻辑.单一来源,常量、魔术字符串, 数据库 Schema 必须定义在集中配置文件中.不要把不需要也不能修改的数据/模型契约、产物文件名定义在配置文件中，如产物保存位置，wing/fuselage部件类型名。
避免为了 DRY 而引入过度复杂的泛型或多层继承。如果消除重复会导致代码可读性急剧下降，请优先选择代码的清晰度，并辅助以显式注释.

4. Fail-Fast 机制暴露错误
禁止过度防御性编程，不使用 config.get('max_workers', 4)的默认参数，必须让潜在的错误直接通过报错暴露出来

5. 严禁把生成产物写到项目文件夹之外

6. 代码确保可直接无参数运行，如有参数参数字段必须短

7. 严禁同时往内存里塞两个风场文件，同时最多有一个文件，允许同个风场多个训练

## CodeGraph 查询规范
为避免单次输出过长被截断，遵循以下顺序：

1. 先用 `codegraph query "符号名"` 定位符号，不在此步骤读取源码。
2. 再用 `codegraph node "类名.方法名"` 读取单个精确符号的源码；默认一次只研究一个问题或符号。
3. 追踪关系时使用 `codegraph callers` 或 `codegraph callees`，每次只沿一个方向追踪一跳。
4. 需要按文件阅读时，使用 `codegraph node 文件名 --file --offset 起始行 --limit 行数` 分页；每页建议 80 至 150 行。
5. 仅在需要模块级概览时使用 `codegraph explore`，并默认传入 `--max-files 1`；不要将其作为读取完整实现的方式。
6. 出现 `Some file sections were trimmed for size` 或 `output truncated to budget` 时，禁止基于缺失内容推断；改用缺失符号的精确 `node` 查询或更小的文件分页。
7. `step`、`reset` 等通用名称的调用链可能混入其他模块的同名符号；必须以文件位置和返回源码复核。

mistakes:
