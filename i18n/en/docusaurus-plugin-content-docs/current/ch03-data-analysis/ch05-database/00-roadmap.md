---
title: "Database Overview: Why You Still Need Databases After Data Analysis"
sidebar_position: 20
description: "A road map for beginners learning databases for the first time: why you still need databases beyond CSVs and DataFrames, and how to study this chapter without getting confused."
keywords: [database overview, relational database, SQL, sqlite, Pandas and databases]
---

# Database Overview: Why You Still Need Databases After Data Analysis

:::tip A very important learning expectation upfront
The main thread of Chapter 3, Data Analysis and Visualization, is still:

- `NumPy`
- `Pandas`
- Visualization
- Data analysis projects

The database chapter is optional,  
so you do not need to treat it as the main battlefield where everything must be mastered right now.  
But if you want to keep going in these directions later:

- enterprise data analysis
- RAG
- AI Agent
- backend data systems

then databases will show up again very quickly.
:::

## Learning objectives

- Know where this chapter fits within Chapter 3, Data Analysis and Visualization
- Understand why databases are still needed beyond `CSV / Excel / DataFrame`
- Understand what the 4 lessons in this chapter are each helping you learn
- Know the safest order to follow when learning databases for the first time

---

## First, build a map

For beginners, the best way to understand this database chapter is not to "memorize SQL syntax first," but to first see clearly:

![Database elective learning roadmap](/img/course/ch03-database-roadmap-en.png)

So what this chapter really wants to solve is:

- Why data does not always stay quietly in a CSV file
- When data needs to be stored long-term, used by multiple people, and queried with permissions, why the system naturally grows into a database

## Why does Chapter 3, Data Analysis and Visualization, add databases later?

Because for beginners, the more stable order is usually:

1. First learn how to handle one table
2. Then learn how to handle multiple tables
3. Finally understand why those tables are stored long-term in a database

In other words, this database chapter is not here to replace `Pandas`,  
but to tell you:

- In the real world, table data often comes from databases

## A more beginner-friendly overall analogy

You can think of what you learned earlier as:

- moving data onto your own desk for analysis

A database is more like:

- a company-wide data archive room

A table on your desk is certainly convenient,  
but when data starts to:

- grow larger
- be used by multiple people
- need long-term storage
- need fast querying

then you can no longer rely only on scattered files.

## What does each of the 4 lessons in this chapter cover?

| Chapter | What problem should it help you solve most? |
|---|---|
| [5.1 Basics of Relational Databases](./01-relational-db.md) | First understand what databases, tables, primary keys, and foreign keys actually are |
| [5.2 SQL Basics](./02-sql-basics.md) | Learn how to use SQL to query, filter, and aggregate data |
| [5.3 Python Database Operations](./03-python-db.md) | Connect Python, Pandas, and databases for real |
| [5.4 Database Design](./04-db-design.md) | Understand why tables are split up that way and why fields are designed that way |

## The safest order for learning databases for the first time

A more stable order is usually:

1. Start with [Basics of Relational Databases](./01-relational-db.md)  
   First build the concepts of databases and tables.
2. Then learn [SQL Basics](./02-sql-basics.md)  
   First learn how to query, filter, and group.
3. Then learn [Python Database Operations](./03-python-db.md)  
   Connect code with the database.
4. Finally read [Database Design](./04-db-design.md)  
   Learn why tables are organized that way.

## What is most worth remembering the first time you learn databases?

What is most worth remembering is actually just these few sentences:

1. A database is a place for storing tables long-term
2. The core idea of relational databases is that tables can be linked through keys
3. SQL is the language used to talk to databases
4. `Pandas` and SQL are not competing tools; they are often used together

## Common mistakes beginners make

- Jumping straight into memorizing all SQL syntax without first understanding table structure
- Only knowing how to use a database to "store things," but not how to query based on business questions
- Thinking that learning databases means having nothing to do with `Pandas` anymore
- Getting scared when seeing `JOIN`, when in essence it is very similar to `merge`

## What should you be able to do after finishing this chapter?

- Know what scenarios are suitable for databases vs. CSV
- Understand the primary key and foreign key relationships in a simple table
- Write the most basic `SELECT / WHERE / JOIN / GROUP BY`
- Use Python to connect to SQLite and read/write simple data

## What you should take away from this chapter

- A database is not "another completely unrelated topic"; it is a natural extension of real data systems
- Understanding tables, keys, and relationships first will make learning SQL much more stable
- The goal of this chapter is not to become a DBA, but to help you stop feeling uneasy when you see a database

## How beginners and advanced learners should read this chapter

When beginners study this chapter for the first time, focus first on the main line and the smallest runnable examples. You do not need to understand every detail at once. As long as you can explain what problem this chapter solves, what the inputs and outputs are, and how the smallest project runs, you can keep moving forward.

Experienced learners can use this chapter for gap-filling and engineering practice: pay attention to edge cases, failure cases, evaluation methods, code reproducibility, and how it connects to the earlier and later stages. After reading, it is best to distill the chapter into your own project README or experiment notes.

## Suggested study time and difficulty

| Study mode | Suggested time | Goal |
|---|---|---|
| Quick overview | 20–30 minutes | Understand what problem this chapter solves and where it will be used later |
| Minimum pass | 1–2 hours | Run a minimal example and complete the chapter's small project exit task |
| Deep practice | Half a day to 1 day | Add error analysis, comparison experiments, or project README notes |

## Chapter self-check questions

| Self-check question | Passing standard |
|---|---|
| What problem does this chapter solve? | You can explain its role in the whole course in one sentence |
| What are the minimum input and output? | You can clearly explain what the example needs as input and what result it produces |
| Where are the common failure points? | You can list at least one reason for an error, poor result, or misunderstanding |
| What can you distill after learning it? | You can write the chapter outcome into a project README, experiment notes, or portfolio |
## Chapter mini-project exit task

After finishing this chapter, it is recommended that you complete a minimum practice task: choose one of the core concepts or tools in this chapter and create a small result that can run, can be screenshotted, and can be written into a README. It does not need to be complex, but it should clearly show what the input is, what the processing steps are, and what the output is.

## Pass criteria

By the end of this chapter, you should be able to explain in your own words what problem this chapter solves, how it relates to the learning stops before and after it, and complete the minimum version of the chapter mini-project exit task.

If you can also record one common mistake, one debugging process, or one result improvement, then it means you are no longer just "reading the content"—you are turning this chapter into your own project experience.
