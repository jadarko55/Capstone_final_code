-- SEDE (Stack Exchange Data Explorer) query
-- Purpose: Extract SQL-tagged questions that look error-related and have accepted answers,
--          along with the accepted answer text.
--
-- How to use:
-- 1) Open SEDE for Stack Overflow: https://data.stackexchange.com/stackoverflow/query/new
-- 2) Paste this query, adjust TOP N if desired, click "Run Query".
-- 3) Export results as CSV and save into data/sql_data/ as QueryResults(1).csv (or update src/config_sql.py).

-- Find SQL error questions with accepted answers
SELECT TOP 10
    q.Id AS QuestionId,
    q.Title,
    q.Body AS QuestionBody,
    q.Score AS QuestionScore,
    q.CreationDate,
    q.Tags,
    q.AcceptedAnswerId,
    q.AnswerCount,
    a.Id AS AnswerId,
    a.Body AS AnswerBody,
    a.Score AS AnswerScore
FROM Posts q
INNER JOIN Posts a ON q.AcceptedAnswerId = a.Id
WHERE 
    -- Question must have SQL tag
    q.Tags LIKE '%<sql>%'
    
    -- Must have accepted answer
    AND q.AcceptedAnswerId IS NOT NULL
    
    -- Filter for error-related questions
    AND (
        q.Title LIKE '%error%' OR
        q.Title LIKE '%incorrect%' OR
        q.Title LIKE '%fix%' OR
        q.Title LIKE '%wrong%' OR
        q.Title LIKE '%syntax%' OR
        q.Body LIKE '%error%'
    )
    
    -- Reasonable quality threshold
    AND q.Score >= 0
    AND a.Score >= 0
    
    -- Post type: 1 = Question, 2 = Answer
    AND q.PostTypeId = 1
    AND a.PostTypeId = 2

ORDER BY q.Score DESC, q.CreationDate DESC;
