
with base as (

    select * 
    from "fivetran_log"."public"."usage_cost"
),

fields as (
    
    select 
        destination_id,
        measured_month,
        amount as dollars_spent
    from base
)

select * 
from fields

