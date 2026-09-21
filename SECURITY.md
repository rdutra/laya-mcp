# Security policy

## Scope

`laya-mcp` is a local decision server. It does not execute commands, access files,
call cloud LLMs, or take autonomous actions. Decisions are advisory signals and are
not authorization, safety, legal, medical, or financial judgments.

## Reporting a vulnerability

Please do not disclose a suspected vulnerability in a public issue. Use the private
security reporting mechanism provided by the repository host, including a concise
description, affected version/commit, reproduction steps, and impact. If private
reporting is unavailable, contact the maintainers before public disclosure.

Do not include model caches, credentials, private source code, or other sensitive
artifacts in a report.

## Operational guidance

Run the stdio server under the intended local user and review MCP client
configuration. Treat model outputs and confidence values as untrusted advisory data;
do not use them as the sole basis for destructive or security-sensitive actions.
