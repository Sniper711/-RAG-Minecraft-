# NVIDIA AI-AGENT 夏季訓練營  
項目名稱：AI-AGENT 夏季訓練營 — RAG 智能對話機器人  
報告日期：2024年8月18日  
項目負責人：[林耀南, SNIPER711 LIN]  


# 項目概述 - 專案：建立RAG對話機器人查各種 Minecraft 密技
對初學 Minecraft 的遊戲玩家們來說, 最頭(有)痛(趣)的是, Minecraft 那無數的密技與攻略.  
網路上琳瑯滿目的 Minecraft 攻略網站, 找起來真的費力, 得要猜中搜尋關鍵字, 還要花時間閱讀比對.  
若能自動爬文, 把網上的遊戲 walkthroughs 收集抓來做成 RAG, 再以大型語言模型 LLM 做成問答機器人, 對遊戲玩家會很方便, 對遊戲公司想推廣自己的遊戲也很有利.  

<img width="500" alt="截圖 2024-08-17 19 31 56" src="https://github.com/user-attachments/assets/89112496-3a8c-4a97-88b7-36267297daab">   [image credit](https://www.reddit.com/r/Minecraftbuilds/comments/sk7hum/here_are_4_different_end_portal_designs_i_came_up/)


# 技術方案與實施步驟
* 服務選擇：首推 NVIDIA NIM 服務. 有玩過 LLM 模型的就知道, 到處申請 Keys 有多不方便. 而在 NVIDIA NIM 裡面, 只要一隻 NVIDIA API KEY 就能選用大部分的推理模型, 例如：
  * 微軟 phi-3-medium-128k-Instruct
  * Meta llama-3-405b-Instruct, llama-3.1-70b-Instruct, llama-3.1-8b-Instruct
  * NVIDIA nemotron-4-340b-reward, nemotron-codstral-7b-v0.1
  * NV-Mistralai mistral-nemo-12b-Instruct
  * Mistralai mathstral-7b-v0.1, mamba-codstral-7b-v0.1, codestral-22b-Instruct-v0.1
  * Google gemma-2-2b-it, shieldgemma-9b
  * Writer palmyra-fin-70b-32k, palmyra-med-70b, palmyra-med-70b-32k
  * Rakuten rakutenai-7b-Instruct, rakutenai-7b-chat
  * Thudm chatglm3-6b
  * Baichuan-int baichuan2-13b-chat  
除了以上的推理模型 Reasoning, 在下圖左邊分類上, 還有視覺 Vision、視覺設計 Vision Design、檢索 Retrival、語音 Speech、生物學 Biology、模擬 Simulation、遊戲 Gaming、醫療 Healthcare、工業 Industral 的模型與SDK能任選吃到飽. 其中還有不少是提供 docker images, 如入寶山一樣.  
[NVIDIA NIM 服務網址, 這次才知道對所有人開放, 用 email 註冊即刻能用](https://build.nvidia.com/explore/reasoning)
  <img width="1000" alt="截圖 2024-08-17 21 19 23" src="https://github.com/user-attachments/assets/374ebab8-290e-4252-abce-6b2a7875bdb6">

* 模型選擇：
  * 我這次 LLM 模型用 ai-mixtral-8x7b-instruct, 這是一個預訓練的混合專家 (Mixture of Experts, MoE) 模型, 在一開始還沒有想好題目要做自然語言處理、圖像辨識、還是邏輯推理的時候, 模型選 ai-mixtral-8x7b-instruct 是個安全的選擇.  
  * RAG 
*	模型选择（必写）： 详细描述项目采用的技术方案，包括大模型的选择理由、RAG模型的优势分析。

*	数据的构建（必写）： 说明数据构建过程、向量化处理方法及其优势。

*	功能整合（进阶版RAG必填）：  介绍进阶的语音功能、Agent功能、多模态等功能的整合策略与实现方法。


# 实施步骤：
*	环境搭建（必写）： 描述开发环境的搭建过程，包括必要的软件、库的安装与配置。
*	代码实现（必写）： 列出关键代码的实现步骤，可附上关键代码截图或代码块。
*	测试与调优： 描述测试过程，包括测试用例的设计、执行及性能调优。
*	集成与部署： 说明各模块集成方法及最终部署到实际运行环境的步骤。


# 项目成果与展示：
*	应用场景展示(必写)： 描述对话机器人的具体应用场景，如客户服务、教育辅导等。
*	功能演示（必写）： 列出并展示实现的主要功能，附上UI页面截图，直观展示项目成果。

# 问题与解决方案：
*	问题分析： 详细描述在项目实施过程中遇到的主要问题。
*	解决措施： 阐述针对每个问题采取的具体解决措施及心路历程，体现问题解决能力。


# 项目总结与展望：
*	项目评估： 对项目的整体表现进行客观评估，总结成功点和存在的不足。
*	未来方向： 基于项目经验，提出未来可能的改进方向和发展规划。


# 附件与参考资料

[列出项目报告中引用的所有附件和参考资料。]
