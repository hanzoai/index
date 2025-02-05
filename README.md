# Hanzo Chat

An AI-powered search engine with a generative UI.

![capture](/public/screenshot-2025-01-31.png)

## 🗂️ Overview

- 🛠 [Features](#-features)
- 🧱 [Stack](#-stack)
- 🚀 [Quickstart](#-quickstart)
- 🌐 [Deploy](#-deploy)
- 🔎 [Search Engine](#-search-engine)
- ✅ [Verified models](#-verified-models)
- ⚡ [AI SDK Implementation](#-ai-sdk-implementation)
- 📦 [Open Source vs Cloud Offering](#-open-source-vs-cloud-offering)
- 👥 [Contributing](#-contributing)

## 🛠 Features

### Core Features

- AI-powered search with GenerativeUI
- Natural language question understanding
- Multiple search providers support (Tavily, SearXNG, Exa)
- Model selection from UI (switch between available AI models)
  - Reasoning models with visible thought process

### Chat & History

- Chat history functionality (Optional)
- Share search results (Optional)
- Redis support (Local/Upstash)

### AI Providers

- OpenAI (Default)
- Google Generative AI
- Azure OpenAI
- Anthropic
- Ollama
- Groq
- DeepSeek
- Fireworks
- OpenAI Compatible

### Search Capabilities

- URL-specific search
- Video search support (Optional)
- SearXNG integration with:
  - Customizable search depth (basic/advanced)
  - Configurable engines
  - Adjustable results limit
  - Safe search options
  - Custom time range filtering

### Additional Features

- Docker deployment ready
- Browser search engine integration

## 🧱 Stack

### Core Framework

- [Next.js](https://nextjs.org/) - App Router, React Server Components
- [TypeScript](https://www.typescriptlang.org/) - Type safety
- [Vercel AI SDK](https://sdk.vercel.ai/docs) - Text streaming / Generative UI

### AI & Search

- [OpenAI](https://openai.com/) - Default AI provider (Optional: Google AI, Anthropic, Groq, Ollama, Azure OpenAI, DeepSeek, Fireworks)
- [Tavily AI](https://tavily.com/) - Default search provider
- Alternative providers:
  - [SearXNG](https://docs.searxng.org/) - Self-hosted search
  - [Exa](https://exa.ai/) - Neural search

### Data Storage

- [Upstash](https://upstash.com/) - Serverless Redis
- [Redis](https://redis.io/) - Local Redis option

### UI & Styling

- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS framework
- [shadcn/ui](https://ui.shadcn.com/) - Re-usable components
- [Radix UI](https://www.radix-ui.com/) - Unstyled, accessible components
- [Lucide Icons](https://lucide.dev/) - Beautiful & consistent icons

## 🚀 Quickstart

### 1. Fork and Clone repo

Fork the repo to your Github account, then run the following command to clone the repo:

```bash
git clone git@github.com:[YOUR_GITHUB_ACCOUNT]/chat.git
```

### 2. Install dependencies

```bash
cd chat
bun install
```

### 3. Configure environment variables

```bash
cp .env.local.example .env.local
```

Fill in the required environment variables in `.env.local`:

```bash
# Required
OPENAI_API_KEY=     # Get from https://platform.openai.com/api-keys
TAVILY_API_KEY=     # Get from https://app.tavily.com/home
```

For optional features configuration (Redis, SearXNG, etc.), see [CONFIGURATION.md](./docs/CONFIGURATION.md)

### 4. Run app locally

#### Using Bun

```bash
bun dev
```

#### Using Docker

```bash
docker compose up -d
```

Visit http://localhost:3000 in your browser.

## 🌐 Deploy

Host your own live version of Hanzo Chat with Vercel or Cloudflare Pages.

### Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fhanzoai%2Fchat&env=OPENAI_API_KEY,TAVILY_API_KEY,UPSTASH_REDIS_REST_URL,UPSTASH_REDIS_REST_TOKEN)

## 🔎 Search Engine

### Setting up the Search Engine in Your Browser

If you want to use Hanzo Chat as a search engine in your browser, follow these steps:

1. Open your browser settings.
2. Navigate to the search engine settings section.
3. Select "Manage search engines and site search".
4. Under "Site search", click on "Add".
5. Fill in the fields as follows:
   - **Search engine**: Hanzo Chat
   - **Shortcut**: hanzo-chat
  - **URL with %s in place of query**: `https://hanzo.ai/search?q=%s`
6. Click "Add" to save the new search engine.
7. Find "Hanzo Chat" in the list of site search, click on the three dots next to it, and select "Make default".

This will allow you to use Hanzo Chat as your default search engine in the browser.

## ✅ Verified models

### List of models applicable to all

- OpenAI
  - o3-mini
  - gpt-4o
  - gpt-4o-mini
  - gpt-4-turbo
  - gpt-3.5-turbo
- Google
  - Gemini 2.0 Pro (Experimental)
  - Gemini 2.0 Flash Thinking (Experimental)
  - Gemini 2.0 Flash
- Anthropic
  - Claude 3.5 Sonnet
  - Claude 3.5 Hike
- Ollama
  - qwen2.5
  - deepseek-r1
- Groq
  - deepseek-r1-distill-llama-70b
- DeepSeek
  - DeepSeek V3
  - DeepSeek R1

## ⚡ AI SDK Implementation

### Current Version: AI SDK UI

This version of Hanzo Chat uses the AI SDK UI implementation, which is recommended for production use. It provides better streaming performance and more reliable client-side UI updates.

### Previous Version: AI SDK RSC (v0.2.34 and earlier)

The React Server Components (RSC) implementation of AI SDK was used in versions up to [v0.2.34](https://github.com/hanzoai/chat/releases/tag/v0.2.34) but is now considered experimental and not recommended for production. If you need to reference the RSC implementation, please check the v0.2.34 release tag.

> Note: v0.2.34 was the final version using RSC implementation before migrating to AI SDK UI.

For more information about choosing between AI SDK UI and RSC, see the [official documentation](https://sdk.vercel.ai/docs/getting-started/navigating-the-library#when-to-use-ai-sdk-rsc).

## 📦 Open Source vs Cloud Offering

Hanzo Chat is open source software available under the Apache-2.0 license.

To maintain sustainable development and provide cloud-ready features, we offer a hosted version of Hanzo Chat alongside our open-source offering. The cloud solution makes Hanzo Chat accessible to non-technical users and provides additional features while keeping the core functionality open and available for developers.

For our cloud service, visit [hanzo.ai](https://hanzo.ai).

## 👥 Contributing

We welcome contributions to Hanzo Chat! Whether it's bug reports, feature requests, or pull requests, all contributions are appreciated.

Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- How to submit issues
- How to submit pull requests
- Commit message conventions
- Development setup
