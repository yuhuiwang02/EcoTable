with issue as (
    select *
    from "github"."public_github_dev"."stg_github__issue"
), 

issue_merged as (
    select
      issue_id,
      min(merged_at) as merged_at
      from "github"."public_github_dev"."stg_github__issue_merged"
    group by 1
)



, pull_request_review as (
    select *
    from "github"."public_github_dev"."stg_github__pull_request_review"
), 


pull_request as (
    select *
    from "github"."public_github_dev"."stg_github__pull_request"
), 

requested_reviewer_history as (
    select *
    from "github"."public_github_dev"."stg_github__requested_reviewer_history"
    where not removed
), 

first_request_time as (
    select
      pull_request.issue_id,
      pull_request.pull_request_id,
      -- Finds the first review that is by the requested reviewer and is not a dismissal
      min(case when requested_reviewer_history.requested_id = pull_request_review.user_id then
          case when lower(pull_request_review.state) in ('commented', 'approved', 'changes_requested') 
                then pull_request_review.submitted_at end 
      else null end) as time_of_first_requested_reviewer_review,
      min(requested_reviewer_history.created_at) as time_of_first_request,
      min(pull_request_review.submitted_at) as time_of_first_review_post_request
    from pull_request
    left join requested_reviewer_history on requested_reviewer_history.pull_request_id = pull_request.pull_request_id
    left join pull_request_review on pull_request_review.pull_request_id = pull_request.pull_request_id
      and pull_request_review.submitted_at > requested_reviewer_history.created_at
    group by 1, 2
)

select
  first_request_time.issue_id,
  issue_merged.merged_at,
  
        (
        (
        (
        ((coalesce(first_request_time.time_of_first_review_post_request, now()))::date - (first_request_time.time_of_first_request)::date)
     * 24 + date_part('hour', (coalesce(first_request_time.time_of_first_review_post_request, now()))::timestamp) - date_part('hour', (first_request_time.time_of_first_request)::timestamp))
     * 60 + date_part('minute', (coalesce(first_request_time.time_of_first_review_post_request, now()))::timestamp) - date_part('minute', (first_request_time.time_of_first_request)::timestamp))
     * 60 + floor(date_part('second', (coalesce(first_request_time.time_of_first_review_post_request, now()))::timestamp)) - floor(date_part('second', (first_request_time.time_of_first_request)::timestamp)))
    / 60/60 as hours_request_review_to_first_review,
  
        (
        (
        (
        ((least(
                            coalesce(first_request_time.time_of_first_requested_reviewer_review, now()),
                            coalesce(issue.closed_at, now())))::date - (first_request_time.time_of_first_request)::date)
     * 24 + date_part('hour', (least(
                            coalesce(first_request_time.time_of_first_requested_reviewer_review, now()),
                            coalesce(issue.closed_at, now())))::timestamp) - date_part('hour', (first_request_time.time_of_first_request)::timestamp))
     * 60 + date_part('minute', (least(
                            coalesce(first_request_time.time_of_first_requested_reviewer_review, now()),
                            coalesce(issue.closed_at, now())))::timestamp) - date_part('minute', (first_request_time.time_of_first_request)::timestamp))
     * 60 + floor(date_part('second', (least(
                            coalesce(first_request_time.time_of_first_requested_reviewer_review, now()),
                            coalesce(issue.closed_at, now())))::timestamp)) - floor(date_part('second', (first_request_time.time_of_first_request)::timestamp)))
     / 60/60 as hours_request_review_to_first_action,
  
        (
        (
        (
        ((merged_at)::date - (first_request_time.time_of_first_request)::date)
     * 24 + date_part('hour', (merged_at)::timestamp) - date_part('hour', (first_request_time.time_of_first_request)::timestamp))
     * 60 + date_part('minute', (merged_at)::timestamp) - date_part('minute', (first_request_time.time_of_first_request)::timestamp))
     * 60 + floor(date_part('second', (merged_at)::timestamp)) - floor(date_part('second', (first_request_time.time_of_first_request)::timestamp)))
    / 60/60 as hours_request_review_to_merge
from first_request_time
join issue on first_request_time.issue_id = issue.issue_id
left join issue_merged on first_request_time.issue_id = issue_merged.issue_id