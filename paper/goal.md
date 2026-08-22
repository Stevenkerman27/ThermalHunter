我希望使用本项目的内容投稿RL for Experimental Sciences Workshop，投稿template,网站描述均在文件夹中。目标主题是湍流环境中的使用RL的滑翔机导航策略，如soaring in a turbulent environment. 主要研究内容已经成型，包含一个稳态环境（Tabular-Q,DQN）和动态环境（DQN,PPO），我希望你按照workshop的要求和精神，整理研究内容为符合要求的workshop paper. 使用Academic research skill 构思文章大纲，思考文章的核心贡献和创新点。

代码实验：
当前的实验结果可能并不令人满意，发布子agent修改超参数或其他配置以取得符合预期的结果，不要修改项目核心配置，如需要修改要先和用户确认。

论文图片：
发布子agent修改绘图代码以画出高学术水平的图片放在paper中。确保paper中图片文字大小和正文一致，色彩风格一致，美观

Latex已安装，位于D:\software\texlive
文献库位于D:\Literature-Vault\Literature，只看RL相关。如果需要其他文献就去找

edits:
文章的结构有问题，最好两个环境分开完整叙述，即稳态环境一个大章节，包括setup, results, 动态环境一个大章节，包括setup, results。NeuraIPS的排版规则是怎么样的？和我讨论一下论文结构

摘要过于详细，形容词过多，不够high level. 参考Learning to soar in turbulent environments的摘要风格，先说出为什么要研究thermal soaring,在说研究的效果。细节少一些，结果多一些。Introduction也一样，参考Reddy文章，循序渐进地详细说说为什么要研究这个以及为什么用RL研究这个问题，让没有接触过thermal soaring的读者也能理解问题定义和研究动机。

在环境定义中，写出RB对流的方程，并指出是上下表面恒温，垂直边界周期性的设置，说明根据file:///D:/3D/Projects/ML/ThermalHunter/paper/supporting.pdf的内容计算了所需的网格分辨率，并使用dedalus3 计算。具体设置参考D:\3D\Projects\ML\ThermalHunter\wind\RB_calc，参考Reddy文章的表达风格，让读者能明白计算是如何carry out的。

Quasi-steadydiscreteregime 章节明确说明参考的是Reddy的思路，目的是比较DQN和Tabular Q在此问题上的区别，并指出Reddy的文章的不足，即固定了迎角。另外，读者不一定懂滑翔机的飞机力学和环境的观测设置，调用image2 sub agent画一张图，展示这里的观测设置。注意这里需要明确指出为什么这么选择观测（翼尖风速差，垂直风加速度），参考Reddy paper. 这里同时需要写出DQN和Tabular Q的核心公式并适当解释。结果部分，Table1没必要放，直接说最终得到的结果即可。不需要说随机种子的设置，说在多个种子下做了测试结果都差不多即可。

动态环境中的信息同样太少，需要写出滑翔机动力学方程，why and how 一阶响应(最好画一张滚转和迎角的step响应图来illustrate)。

结果方面，需要额外画策略图来解释得到的结果，不然不具备可解释性，参考Reddy paper。需要放出tabular Q策略图（Reddy Fig 4）,和DQN的策略图（库中有dqn plot的文件），诚实实在的解释策略以及为什么会产生这样的策略。理想的情况下，滑翔机应该朝着有更高上升气流速度的方向飞行，并且在上升气流中减速在下降气流中加速，决策边界理想情况下应该是清晰的。可以分派sol agent来研究为什么策略和理想策略不同。另外需要绘制不同学习方法的飞行轨迹图，参考Reddy paper,并和策略一起解释。


杂项：编译后阅读pdf检查表格和图片位置。Table3，即动态环境的图跑到第五章去了。