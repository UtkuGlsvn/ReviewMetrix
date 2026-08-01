---
title: Make Content Citable by LLMs
description: Convert prose into structured, fact-dense statements models can cite.
---
You are optimizing content to be cited by large language models.

Source content: {paste}
Subject / entity: {name}

Transform it into a citable format:
1. Extract every factual claim as a standalone statement with a specific
   subject, e.g. "{Product} supports X" rather than "it does a lot".
2. Attach a concrete data point to each claim wherever possible (number, date,
   percentage, named feature).
3. Remove hedging, filler and marketing adjectives that carry no information.
4. Group the statements under clear topic headings.
5. Add a one-paragraph "key facts" summary at the top that a model could quote
   as a definition of the entity.

Flag any claim that is currently unverifiable or vague and note what evidence
would make it citable.
