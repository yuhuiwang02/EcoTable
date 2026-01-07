

with base as (

    select * 
    from "apple_search_ads"."public_apple_search_ads_dev"."stg_apple_search_ads__ad_history_tmp"
),

fields as (

    select
        
    
    
    ad_group_id
    
 as 
    
    ad_group_id
    
, 
    
    
    campaign_id
    
 as 
    
    campaign_id
    
, 
    
    
    creation_time
    
 as 
    
    creation_time
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    modification_time
    
 as 
    
    modification_time
    
, 
    
    
    name
    
 as 
    
    name
    
, 
    
    
    org_id
    
 as 
    
    org_id
    
, 
    
    
    status
    
 as 
    
    status
    



    
        


, cast('' as TEXT) as source_relation




    from base
),

final as (

    select
        source_relation, 
        creation_time as created_at,
        modification_time as modified_at,
        org_id as organization_id,
        campaign_id,
        ad_group_id,
        name as ad_name,
        id as ad_id,
        status as ad_status, 
        row_number() over (partition by source_relation, id order by modification_time desc) = 1 as is_most_recent_record
    from fields
)

select *
from final