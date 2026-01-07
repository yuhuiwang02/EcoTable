

with base as (
    
    select * from "fivetran_log"."public"."destination_membership"
),

fields as (

    select
        destination_id,
        user_id,
        cast(activated_at as timestamp) as activated_at,
        cast(joined_at as timestamp) as joined_at,
        role as destination_role
    from base
)

select * 
from fields