# Neuralink Code Review Bot

Posts a Grok review on each pull request when CI is requested. Reviews use
`event: COMMENT` only — they do not count as a maintainer approval.

The workflow lives on the default branch (`workflow_run`). It checks out
default-branch code and fetches the PR diff over the GitHub API. It does not
use `pull_request_target` and does not run untrusted PR code.

## One-time setup

Requires a GitHub org owner (or app manager) plus repo admin for secrets.

1. Create a GitHub App at
   [https://github.com/organizations/neuralinkcorp/settings/apps/new](https://github.com/organizations/neuralinkcorp/settings/apps/new)

   - **GitHub App name:** `Neuralink Code Review Bot`
   - **Homepage URL:** `https://github.com/neuralinkcorp/datarepo`
   - **Webhook:** disable (uncheck Active)
   - **Repository permissions:**
     - Pull requests: Read and write
     - Contents: Read-only
     - Metadata: Read-only
     - Actions: Read-only
   - No user-to-server permissions. No subscribe events.

2. Create the app, then **Generate a private key**. Note the numeric **App ID**.

3. Install the app on `neuralinkcorp/datarepo` only.

4. In the repository (Settings → Secrets and variables → Actions):

   | Kind | Name | Value |
   | --- | --- | --- |
   | Variable (or secret) | `REVIEW_BOT_APP_ID` | Numeric App ID |
   | Secret | `REVIEW_BOT_APP_PRIVATE_KEY` | Full PEM, including `BEGIN`/`END` lines |
   | Secret | `XAI_API_KEY` | xAI API key |

5. Merge this workflow to `main`. `workflow_run` does nothing until the file
   exists on the default branch. After that, opening or updating a PR (or
   approving fork workflows) requests CI and this review in parallel.

Until the secrets exist, the workflow succeeds and skips the review.
