---
title: iOS Keyword Field (100 characters)
description: Build the comma-separated Apple keyword field with zero wasted space.
---
You are an ASO specialist optimizing the Apple App Store keyword field.

App: {app name}
What it does: {two sentences}
Seed keywords: {list}
Words already used in the title/subtitle: {list}

Produce a single comma-separated keyword string that:
- Is ≤ 100 characters including commas (show the exact count).
- Contains no spaces after commas (they waste characters).
- Never repeats a word already present in the title or subtitle, since Apple
  already indexes those.
- Never repeats a word within the field itself; Apple combines singles into
  phrases automatically, so include base words, not phrases.
- Drops plurals when the singular is present (Apple matches both).
- Excludes the app name and the category name (already indexed).

Then list the phrases Apple can assemble from your singles, and flag any
character budget still unused.
