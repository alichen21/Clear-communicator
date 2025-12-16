# Clear Communicator - 清晰表达

一个基于 AI 的表达质量分析工具，帮助用户提升口头表达能力。

## 功能特性

- 🎤 **一键录音**：最长 5 分钟录音，自动转写
- 📊 **四维度分析**：
  - 表达指向性（自我 vs 他人）
  - 结构清晰度（核心能力）
  - 具体程度（Concrete & Specific）
  - 主题与重点（Key Message）
- 💡 **框架推荐**：当结构清晰度较弱时，推荐使用表达框架（过去-现在-未来、空-雨-伞、3C）
- 📝 **历史记录**：ChatGPT 风格的分割布局，记录所有练习历史
- 🔄 **迭代练习**：支持在同一主题上多次录音，比较改进

## 技术栈

- **后端**：FastAPI + Python
- **前端**：HTML + CSS + JavaScript
- **AI API**：AI Builders Space Backend

## 部署

项目已配置 Dockerfile，支持部署到 Koyeb 平台。

### 环境变量

- `AI_BUILDER_TOKEN` 或 `SUPER_MIND_API_KEY`：API 密钥
- `PORT`：服务器端口（部署时自动设置）

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

4. 访问：http://localhost:8000

