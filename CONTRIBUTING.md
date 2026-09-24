# Contributing Guide

Thanks for considering a contribution! This project is meant to stay
beginner-friendly, so the bar for contributing is intentionally low.

## Ground rules

1. **Keep it free.** Don't add a dependency on a paid API, paid tier, or
   anything that requires a credit card. If you want to add support for
   a new LLM provider or vector store, it must have a genuinely free
   local or self-hosted option.
2. **Keep it explainable.** Every non-obvious decision should have a
   comment explaining *why*, not just *what*. Imagine explaining it to
   someone who has never seen this codebase.
3. **Small, focused changes.** One feature or fix per pull request is
   easier to review and easier to undo if something goes wrong.

## How to contribute

1. **Fork** this repository (click "Fork" on GitHub).
2. **Clone your fork** to your computer:
   ```bash
   git clone https://github.com/YOUR-USERNAME/semantic-cache-llm.git
   cd semantic-cache-llm
   ```
3. **Create a branch** for your change:
   ```bash
   git checkout -b my-improvement
   ```
4. **Make your changes**, and add or update tests in `tests/` if the
   behavior changed.
5. **Run the test suite** before opening a pull request:
   ```bash
   pytest
   ```
6. **Commit and push:**
   ```bash
   git add .
   git commit -m "Describe what you changed and why"
   git push origin my-improvement
   ```
7. **Open a Pull Request** on GitHub and describe:
   - What you changed
   - Why you changed it
   - How you tested it

## Reporting bugs / suggesting ideas

Open a GitHub Issue. Include:
- What you expected to happen
- What actually happened
- Steps to reproduce (if it's a bug)
- Your OS and Python version

## Code style

- Follow existing formatting (4-space indents, descriptive names).
- Prefer clarity over cleverness — this is a learning project as much
  as a working one.
- Add a short comment block at the top of any new file explaining its
  purpose, the way the existing files do.

No contribution is too small — fixing a typo in the README counts!
