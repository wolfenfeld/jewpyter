---
layout: post
title: Who Signs for the Package?
description: |
  Data contracts in the age of agents that write back
noindex: true
---

My nephew Baruch spent a summer "helping" in my mother's kitchen.
He meant well.
He also moved the salt to where the sugar used to be, because in his opinion that made more sense, and he never told anyone.
Three weeks later my aunt made a brisket that could strip paint, and the investigation took longer than the brisket did.

Nobody blamed Baruch, exactly.
He wasn't malicious. He was just a new set of hands in a kitchen with rules nobody had written down, operating on his own idea of where things ought to go.

I bring this up because we spent a decade building data pipelines around the assumption that the people touching our tables were, at minimum, people — colleagues who could be asked a question, who felt the social cost of breaking something and tended not to, who drifted slowly if they drifted at all.
That assumption is now wrong often enough to matter.

# The Old World

A data contract, in the boring sense everyone already knows, is a promise: the orders table will have a `status` column, it will only ever contain one of four values, and if that's about to change, someone will say so before it happens.

The failure mode this protects against is almost always social, not technical.
A producer team changes a field to unblock their own sprint.
They don't loop in whoever's dashboard depends on it, because they don't know that dashboard exists, or they assume someone would have said something by now.
Three weeks later, someone's weekly report is quietly wrong, and nobody notices until finance asks why revenue doubled overnight.

Slow, social, and — this part matters — bounded by how much damage one team's Tuesday afternoon can do.

# What an Agent Changes

An LLM agent with write access to that same table isn't a colleague you can pull aside.
It doesn't drift over a quarter because two teams stopped talking.
It can invent a new column, misread a nullable field as optional-to-fill, or "improve" a value it has decided looks wrong — all in the same afternoon, without anyone in the loop to feel the cost of being wrong.

The failure mode moves from slow-and-social to fast-and-silent.
Baruch, at least, could only rearrange one kitchen. An agent can rearrange every kitchen in the building before anyone's had their coffee.

# The Fix That Sounds Right and Isn't

"So put the schema in the prompt," said my cousin Devorah, who has strong opinions about most things and is annoyingly often right about them.

Not this time.

Telling an agent the shape of a table in its instructions is advisory, not enforced — the same as leaving my mother a note that says *please don't move the salt*. It might work. It relies entirely on the note being read, understood the way you meant it, and followed exactly, by something that has no memory of the last time it didn't.

A schema in a prompt is a request for good behavior.
A data contract was never supposed to be a request.

# Where Enforcement Actually Has to Live

If the promise can't live in a conversation, it has to live at the boundary the agent can't talk its way around — the write path itself.

That's not a new idea. Schema Registry has been rejecting malformed Avro records for years. dbt contracts and Great Expectations have been failing builds on schema drift for years. What's new is who's on the other side of that boundary now, and how much less patient we can afford to be with it.

Treat the agent the way you'd treat any external API caller you didn't write yourself: not as a trusted member of the team who happens to be a bit forgetful, but as an untrusted writer whose every insert or update gets checked against the contract before it lands, no exceptions carved out because "it's probably fine, it's just the agent."

# A Small Example

Say an agent is asked to update a customer's status after a support call. The column has always held one of `active`, `churned`, `paused`.

The agent decides, reasonably from its own vantage point, that the customer hasn't technically churned yet but is clearly heading there, and writes `probably_churned`.

Every downstream dashboard that filters on those three known values just quietly stops counting that customer as anything at all.
Nothing crashed. Nothing alerted. The number is just wrong now, and it will stay wrong until somebody happens to look closely enough to ask why.

A contract enforced at the write boundary doesn't let that insert through in the first place. Not because it understands the agent was well-intentioned — it doesn't need to. It only needs to know that `probably_churned` isn't one of the four values anyone agreed to.

# So What Was the Contract Ever For

Here's what I think we got backwards, talking about this with Devorah until my tea went cold.

We used to describe a data contract as an agreement about a schema. But nobody ever really argued about the schema. What we were actually agreeing on was who got to make judgment calls about what the data meant — and the whole point of the contract was that it wasn't the writer.

Baruch never got to decide the salt belonged somewhere new. That was never his call to make, however sensible his reasoning. It doesn't matter whether the one making that call is a nephew with opinions or a model with none.
