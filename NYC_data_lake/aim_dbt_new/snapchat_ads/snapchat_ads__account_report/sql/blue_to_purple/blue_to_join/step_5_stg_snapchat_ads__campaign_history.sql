

with base as (

    select * 
    from "snapchat_ads"."public_snapchat_ads_dev"."stg_snapchat_ads__campaign_history_tmp"

),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    ad_account_id
    
 as 
    
    ad_account_id
    
, 
    cast(null as timestamp) as 
    
    created_at
    
 , 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    name
    
 as 
    
    name
    
, 
    cast(null as timestamp) as 
    
    updated_at
    
 , 
    cast(null as integer) as 
    
    daily_budget_micro
    
 , 
    cast(null as timestamp) as 
    
    start_time
    
 , 
    cast(null as timestamp) as 
    
    end_time
    
 , 
    cast(null as integer) as 
    
    lifetime_spend_cap_micro
    
 , 
    cast(null as TEXT) as 
    
    status
    
 , 
    cast(null as TEXT) as 
    
    objective
    
 



        


, cast('' as TEXT) as source_relation




    from base

),

final as (

    select
        source_relation,
        id as campaign_id,
        ad_account_id,
        cast(created_at as timestamp) as created_at,
        name as campaign_name,
        cast(_fivetran_synced as timestamp) as _fivetran_synced,
        cast(updated_at as timestamp) as updated_at,
        (daily_budget_micro / 1000000.0) as daily_budget,
        cast(start_time as timestamp) as start_time,
        cast(end_time as timestamp) as end_time,
        (lifetime_spend_cap_micro / 1000000.0) as lifetime_spend_cap,
        status,
        objective,
        row_number() over (partition by source_relation, id order by _fivetran_synced desc) = 1 as is_most_recent_record
    from fields

)

select * 
from final