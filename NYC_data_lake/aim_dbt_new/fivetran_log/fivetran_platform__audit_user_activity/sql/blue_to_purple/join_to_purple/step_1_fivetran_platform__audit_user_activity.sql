with logs as (

    select 
        *,
        

  case when message_data ~ '^\s*[\{].*[\}]?\s*$' -- Postgres has no native json check, so this will check the string for indicators of a JSON object
    then message_data::jsonb #>> '{actor}'
    else null end

 as actor_email
    from "fivetran_log"."public_fivetran_platform"."stg_fivetran_platform__log"
    where lower(message_data) like '%actor%'
),

user_logs as (

    select *
    from logs
    where actor_email is not null 
        and lower(actor_email) != 'fivetran'
),

connection as (

    select *
    from "fivetran_log"."public_fivetran_platform"."stg_fivetran_platform__connection"
),

destination as (

    select *
    from "fivetran_log"."public_fivetran_platform"."stg_fivetran_platform__destination"
),
users as (

    select *
    from "fivetran_log"."public_fivetran_platform"."stg_fivetran_platform__user"
),
    destination_membership as (

        select *
        from "fivetran_log"."public_fivetran_platform"."stg_fivetran_platform__destination_membership"
    ),
    final as (

    select
        date_trunc('day', user_logs.created_at) as date_day,
        
        to_char(user_logs.created_at, 'FMDy') as day_name,
            date_part('day', user_logs.created_at) as day_of_month,
        user_logs.created_at as occurred_at,
        destination.destination_name,
        destination.destination_id,
        connection.connection_name,
        connection.connection_id,
        user_logs.actor_email as email,
        users.first_name,
        users.last_name,
        users.user_id,
        destination_membership.destination_role,
    user_logs.event_type, -- should always be INFO for user-triggered actions but include just in case
        user_logs.event_subtype,
        user_logs.message_data,
        user_logs.log_id

    from user_logs
    left join connection
        on user_logs.connection_id = connection.connection_id
    left join destination
        on connection.destination_id = destination.destination_id
    left join users 
        on lower(users.email) = lower(user_logs.actor_email)
    left join destination_membership
        on destination.destination_id = destination_membership.destination_id
        and users.user_id = destination_membership.user_id

    )

select *
from final