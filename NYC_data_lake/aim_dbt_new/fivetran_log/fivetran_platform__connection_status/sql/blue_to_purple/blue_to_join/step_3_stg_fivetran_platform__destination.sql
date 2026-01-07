with base as (

    select * 
    from "fivetran_log"."public"."destination"
),

fields as (

    select
        id as destination_id,
        account_id,
        cast(created_at as timestamp) as created_at,
        name as destination_name,
        region
    from base
)

select * 
from fields