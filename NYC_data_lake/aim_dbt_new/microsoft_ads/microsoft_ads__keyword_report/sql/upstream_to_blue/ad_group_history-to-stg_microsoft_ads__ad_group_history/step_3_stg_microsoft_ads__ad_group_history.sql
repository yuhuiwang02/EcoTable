

with base as (

    select * 
    from "microsoft_ads"."public_microsoft_ads_dev"."stg_microsoft_ads__ad_group_history_tmp"
),

fields as (

    select
        
    
    
    id
    
 as 
    
    id
    
, 
    
    
    name
    
 as 
    
    name
    
, 
    
    
    campaign_id
    
 as 
    
    campaign_id
    
, 
    
    
    modified_time
    
 as 
    
    modified_time
    
, 
    
    
    start_date
    
 as 
    
    start_date
    
, 
    
    
    end_date
    
 as 
    
    end_date
    
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
        id as ad_group_id,
        name as ad_group_name,
        campaign_id,
        modified_time as modified_at,
        start_date,
        end_date,
        status,
        row_number() over (partition by source_relation, id order by modified_time desc) = 1 as is_most_recent_record
    from fields
)

select * 
from final