

with base as (

    select *
    from "mailchimp"."public_mailchimp_dev"."stg_mailchimp__automation_recipients_tmp"

), 

fields as (

    select 
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    automation_email_id
    
 as 
    
    automation_email_id
    
, 
    
    
    list_id
    
 as 
    
    list_id
    
, 
    
    
    member_id
    
 as 
    
    member_id
    



        
    from base

), 


final as (

     select
        member_id,
        automation_email_id,
        list_id
    from fields

),

 unique_key as (

    select 
        *,
        md5(cast(coalesce(cast(member_id as TEXT), '_dbt_utils_surrogate_key_null_') || '-' || coalesce(cast(automation_email_id as TEXT), '_dbt_utils_surrogate_key_null_') as TEXT)) as automation_recipient_id
    from final

)

select *
from unique_key