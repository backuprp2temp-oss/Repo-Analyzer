import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import Groq from 'groq-sdk';
import { randomUUID } from 'crypto';

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

const sessions = new Map();

async function callAI({ provider, apiKey, messages, temperature = 0.7, max_tokens = 4096 }) {
  if (provider === 'nvidia') {
    const res = await fetch('https://integrate.api.nvidia.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'deepseek-ai/deepseek-v4-pro',
        messages,
        temperature,
        max_tokens
      })
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error?.message || `NVIDIA NIM error: ${res.status}`);
    }
    return data.choices[0]?.message?.content || 'No response.';
  }
  // Default: Groq
  const groq = new Groq({ apiKey });
  const completion = await groq.chat.completions.create({
    messages,
    model: 'openai/gpt-oss-120b',
    temperature,
    max_tokens
  });
  return completion.choices[0]?.message?.content || 'No response.';
}

function parseGitHubUrl(url) {
  const match = url.match(/github\.com\/([^\/]+)\/([^\/]+)/);
  if (!match) return null;
  return { owner: match[1], repo: match[2].replace(/\.git$/, '') };
}

async function fetchGitHubApi(path) {
  const res = await fetch(`https://api.github.com${path}`, {
    headers: {
      'Accept': 'application/vnd.github.v3+json',
      'User-Agent': 'github-repo-analyzer'
    }
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`GitHub API error: ${res.status} ${text}`);
  }
  return res.json();
}

async function fetchRepoData(owner, repo) {
  const repoInfo = await fetchGitHubApi(`/repos/${owner}/${repo}`).catch(() => null);
  if (!repoInfo) {
    throw new Error('Repository not found or not accessible.');
  }

  const defaultBranch = repoInfo.default_branch;

  const [readme, tree] = await Promise.all([
    fetchGitHubApi(`/repos/${owner}/${repo}/readme`).catch(() => null),
    fetchGitHubApi(`/repos/${owner}/${repo}/git/trees/${defaultBranch}?recursive=1`).catch(() => null)
  ]);

  const treeEntries = tree?.tree || [];
  const filePaths = treeEntries
    .filter(e => e.type === 'blob' && e.path)
    .map(e => e.path)
    .filter(p => !p.includes('node_modules/') && !p.includes('.git/') && !p.includes('dist/') && !p.includes('build/'))
    .slice(0, 200);

  const keyFiles = ['package.json', 'requirements.txt', 'Cargo.toml', 'pyproject.toml', 'setup.py', 'go.mod', 'pom.xml', 'build.gradle', 'Makefile', 'Dockerfile', 'README.md', 'tsconfig.json', 'vite.config.ts', 'next.config.js'];
  const configContents = {};

  for (const file of keyFiles) {
    if (filePaths.includes(file)) {
      try {
        const content = await fetchGitHubApi(`/repos/${owner}/${repo}/contents/${file}`);
        if (content.content) {
          configContents[file] = Buffer.from(content.content, 'base64').toString('utf-8').substring(0, 5000);
        }
      } catch (e) {
        // ignore
      }
    }
  }

  let readmeContent = '';
  if (readme?.content) {
    readmeContent = Buffer.from(readme.content, 'base64').toString('utf-8').substring(0, 10000);
  }

  return {
    name: repoInfo.full_name,
    description: repoInfo.description || '',
    language: repoInfo.language || '',
    stars: repoInfo.stargazers_count || 0,
    topics: repoInfo.topics || [],
    fileCount: treeEntries.filter(e => e.type === 'blob').length,
    filePaths,
    readme: readmeContent,
    configFiles: configContents
  };
}

function formatRepoData(data) {
  return `
Repository: ${data.name}
Description: ${data.description}
Primary Language: ${data.language}
Stars: ${data.stars}
Topics: ${data.topics.join(', ')}
Total Files: ${data.fileCount}

README:
${data.readme}

Key Configuration Files:
${Object.entries(data.configFiles).map(([k, v]) => `--- ${k} ---\n${v}`).join('\n\n')}

File Structure (first 200 non-build files):
${data.filePaths.join('\n')}
`;
}

app.post('/api/analyze', async (req, res) => {
  const { repoUrl, apiKey, provider } = req.body;
  if (!repoUrl) return res.status(400).json({ error: 'Missing repoUrl' });
  if (!apiKey) return res.status(400).json({ error: 'Missing apiKey' });

  const parsed = parseGitHubUrl(repoUrl);
  if (!parsed) return res.status(400).json({ error: 'Invalid GitHub URL. Expected format: https://github.com/owner/repo' });

  try {
    const repoData = await fetchRepoData(parsed.owner, parsed.repo);
    const context = formatRepoData(repoData);

    const analysis = await callAI({
      provider,
      apiKey,
      messages: [
        {
          role: 'system',
          content: 'You are an expert software engineer and architect. Analyze the given GitHub repository thoroughly. Provide insights on its architecture, tech stack, code organization, potential improvements, and anything noteworthy. Be concise but thorough. Use markdown formatting where appropriate.'
        },
        {
          role: 'user',
          content: `Please analyze this repository:\n\n${context}`
        }
      ],
      temperature: 0.7,
      max_tokens: 4096
    });

    const sessionId = randomUUID();
    sessions.set(sessionId, {
      repoContext: context,
      provider,
      messages: [
        { role: 'system', content: 'You are an expert software engineer. You have analyzed a GitHub repository. Use the repository context to answer user questions accurately. Repository context:\n\n' + context },
        { role: 'assistant', content: analysis }
      ]
    });

    res.json({ analysis, sessionId });
  } catch (err) {
    console.error('Analyze error:', err);
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/chat', async (req, res) => {
  const { sessionId, message, apiKey, provider } = req.body;
  if (!sessionId || !message) return res.status(400).json({ error: 'Missing sessionId or message' });
  if (!apiKey) return res.status(400).json({ error: 'Missing apiKey' });

  const session = sessions.get(sessionId);
  if (!session) return res.status(404).json({ error: 'Session not found' });

  session.messages.push({ role: 'user', content: message });

  try {
    const reply = await callAI({
      provider,
      apiKey,
      messages: session.messages,
      temperature: 0.7,
      max_tokens: 4096
    });

    session.messages.push({ role: 'assistant', content: reply });

    res.json({ reply });
  } catch (err) {
    console.error('Chat error:', err);
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
