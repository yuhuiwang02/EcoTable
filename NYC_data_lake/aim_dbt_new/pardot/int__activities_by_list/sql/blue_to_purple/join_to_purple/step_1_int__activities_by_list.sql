with prospects as (

    select *
    from "pardot"."public_pardot"."pardot__prospects"

), lists as (

    select *
    from "pardot"."public_stg_pardot"."stg_pardot__list"

), list_membership as (

    select *
    from "pardot"."public_stg_pardot"."stg_pardot__list_membership"

), joined as (

    select 
        list_membership.list_id,
        prospects.count_activity_visits,
        prospects.count_activity_emails,
        prospects.most_recent_visit_activity_timestamp,
        prospects.most_recent_email_activity_timestamp
    from list_membership
    left join prospects
        using (prospect_id)

), aggregated as (

    select 
        list_id,
        sum(count_activity_emails) as count_activity_emails,
        sum(count_activity_visits) as count_activity_visits,
        max(most_recent_email_activity_timestamp) as most_recent_email_activity_timestamp,
        max(most_recent_visit_activity_timestamp) as most_recent_visit_activity_timestamp
    from joined
    group by 1

)

select *
from aggregated