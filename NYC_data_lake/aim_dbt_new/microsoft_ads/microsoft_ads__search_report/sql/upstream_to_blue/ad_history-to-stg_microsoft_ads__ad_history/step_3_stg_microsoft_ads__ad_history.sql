

with base as (

    select * 
    from "microsoft_ads"."public_microsoft_ads_dev"."stg_microsoft_ads__ad_history_tmp"
),

fields as (

    select
        
    
    
    id
    
 as 
    
    id
    
, 
    
    
    title
    
 as 
    
    title
    
, 
    
    
    title_part_1
    
 as 
    
    title_part_1
    
, 
    
    
    title_part_2
    
 as 
    
    title_part_2
    
, 
    
    
    title_part_3
    
 as 
    
    title_part_3
    
, 
    
    
    final_url
    
 as 
    
    final_url
    
, 
    
    
    ad_group_id
    
 as 
    
    ad_group_id
    
, 
    
    
    modified_time
    
 as 
    
    modified_time
    
, 
    
    
    status
    
 as 
    
    status
    
, 
    
    
    type
    
 as 
    
    type
    
, 
    
    
    domain
    
 as 
    
    domain
    



        
    
        


, cast('' as TEXT) as source_relation




    from base
),

final as (

    select
        source_relation, 
        id as ad_id,
        title_part_1 as ad_name,
        title,
        title_part_1,
        title_part_2,
        title_part_3,
        final_url,
        domain,
        ad_group_id,
        modified_time as modified_at,
        status,
        type,
        row_number() over (partition by source_relation, id order by modified_time desc) = 1 as is_most_recent_record
    from fields
)

select * 
from final