

with base as (

    select * 
    from "fivetran_log"."public"."credits_used"
),

fields as (
    
    select 
        destination_id,
        measured_month,
        credits_consumed as credits_spent
    from base
)

select * 
from fields

