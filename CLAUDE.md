# 项目约定

## 新增 Provider
1. 在 app/providers/ 创建新文件，继承 ImageProvider 基类
2. 在 app/providers/__init__.py 的工厂函数中注册
3. 在 app/credits.py 中添加积分定价
4. 在 config.py 的模型列表中注册

## 新增 Pipeline
1. 在 app/pipelines/ 创建新文件
2. 在 app/routes/pipeline_routes.py 注册路由
3. 在前端添加对应 composable

## 分支规范
所有修改提交到 claude/feature-xxx 分支，开 Draft PR
# AI Creator Space — Claude 开发指南

## 项目概览
FastAPI + Vue 3 多模态 AI 创作平台。
后端：app/routes / providers / pipelines
前端：static/js/composables/

## 架构约定

### 新增 AI Provider
1. 在 `app/providers/` 创建新文件，继承 `ImageProvider` 基类
2. 在 `app/providers/__init__.py` 的 `get_provider_for_model()` 工厂函数注册
3. 在 `app/credits.py` 添加积分定价
4. 在 `app/config.py` 的模型列表中注册

### 新增 Pipeline
1. 在 `app/pipelines/` 创建新文件
2. 在 `app/routes/pipeline_routes.py` 注册路由
3. 在 `static/js/composables/` 添加对应 composable
4. 在 `static/index.html` 添加 UI 入口

### 前端新增功能
1. 在 `static/js/composables/` 创建 useXxx.js
2. 在 `static/js/main.js` 引入并注册
3. 样式使用 Tailwind CSS utility class

## 关键文件速查
| 文件 | 职责 |
|------|------|
| app/providers/__init__.py | Provider 基类 + 工厂函数 |
| app/credits.py | 积分定价表 |
| app/config.py | 模型列表、环境变量 |
| app/job_queue.py | 异步任务队列 |
| static/js/main.js | Vue 根组件入口 |

## 分支与 PR 规范
- 分支名：`claude/issue-{issue号}-{简短描述}`
- PR 标题：`[Issue #{号}] 功能描述`
- PR 必须关联 Issue：在描述中写 `Closes #issue号`
- PR 目标分支：`nalaxu/ai-creator` 的 `main`

## 禁止事项
- 不修改 `config.json`（用户配置，不入库）
- 不修改 `.env`（密钥文件）
- 不删除现有 Provider/Pipeline（只新增或修改）