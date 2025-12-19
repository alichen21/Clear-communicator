# 言值 - 让每一次表达都有价值

一个基于 AI 的表达质量分析工具，帮助用户系统提升口头表达能力。表达训练框架 Inspired by 黄执中的表达实战营。

## 功能特性

- 🎤 **一键录音**：最长 3 分钟的语音录制
- 📝 **自动转录**：AI 驱动的语音转文字
- 📊 **四维度分析**：
  - 表达指向性（自我 vs 他人）
  - 结构清晰度（核心能力）
  - 具体程度（Concrete & Specific）
  - 主题与重点（Key Message）
- 🔄 **迭代练习**：同一主题多次尝试，跟踪改进
- 💡 **框架推荐**：当结构清晰度较弱时，推荐表达框架（过去-现在-未来、空-雨-伞、3C）
- 📚 **历史记录**：ChatGPT 风格的分割布局，方便查看历史练习

## 技术栈

- **后端**：FastAPI + Python
- **前端**：HTML + CSS + JavaScript
- **AI 服务**：AI Builders Space Backend API

## 本地运行

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 创建 `.env` 文件：
```
SUPER_MIND_API_KEY=your_api_key_here
```

3. 运行服务器：
```bash
python main.py
# 或
uvicorn main:app --reload
```

4. 访问：
   - Landing 页面：http://localhost:8000
   - 主应用页面：http://localhost:8000/app

## 部署

本项目支持部署到 ai-builders.space 平台。

## License

MIT
