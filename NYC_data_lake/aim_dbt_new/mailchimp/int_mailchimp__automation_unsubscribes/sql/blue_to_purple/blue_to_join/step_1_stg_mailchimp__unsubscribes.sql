

with base as (

    select * 
    from "mailchimp"."public_mailchimp_dev"."stg_mailchimp__unsubscribes_tmp"

),


fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    campaign_id
    
 as 
    
    campaign_id
    
, 
    
    
    list_id
    
 as 
    
    list_id
    
, 
    
    
    member_id
    
 as 
    
    member_id
    
, 
    
    
    reason
    
 as 
    
    reason
    
, 
    
    
    timestamp
    
 as 
    
    timestamp
    



        
    from base

), 

final as (

    select 
        campaign_id,
        member_id,
        list_id,
        reason as unsubscribe_reason,
        timestamp as unsubscribe_timestamp
    from fields

), 

unique_key as (

    select 
        *, 
        md5(cast(coalesce(cast(member_id as TEXT), '_dbt_utils_surrogate_key_null_') || '-' || coalesce(cast(list_id as TEXT), '_dbt_utils_surrogate_key_null_') || '-' || coalesce(cast(unsubscribe_timestamp as TEXT), '_dbt_utils_surrogate_key_null_') as TEXT)) as unsubscribe_id
    from final

)

select *
from unique_key