

with base as (

    select * 
    from "fivetran_log"."public"."transformation_runs"
),

fields as (

    select
        _fivetran_synced,
        destination_id,
        upper(free_type) as free_type,
        job_id,
        job_name,
        cast(measured_date as timestamp) as measured_date,
        model_runs,
        project_type,
        updated_at
    from base
)

select
    *,
    cast(date_trunc('month', measured_date) as date) as measured_month
from fields

