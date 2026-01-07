

with base as (

    select * 
    from "fivetran_log"."public"."user"
),

fields as (

    select
        id as user_id,
        cast(created_at as timestamp) as created_at,
        email,
        email_disabled as has_disabled_email_notifications,
        family_name as last_name,
        given_name as first_name,
        phone,
        verified as is_verified
    from base
)

select * 
from fields