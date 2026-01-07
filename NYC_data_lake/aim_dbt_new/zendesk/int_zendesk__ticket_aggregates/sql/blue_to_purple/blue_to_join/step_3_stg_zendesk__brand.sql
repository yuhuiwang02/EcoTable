

with base as (

    select * 
    from "google_ads"."public_zendesk_dev"."stg_zendesk__brand_tmp"

),

fields as (

    select
        /*
        The below macro is used to generate the correct SQL for package staging models. It takes a list of columns 
        that are expected/needed (staging_columns from dbt_zendesk/models/tmp/) and compares it with columns 
        in the source (source_columns from dbt_zendesk/macros/).
        For more information refer to our dbt_fivetran_utils documentation (https://github.com/fivetran/dbt_fivetran_utils.git).
        */
        
    
    
    _fivetran_deleted
    
 as 
    
    _fivetran_deleted
    
, 
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    active
    
 as 
    
    active
    
, 
    
    
    brand_url
    
 as 
    
    brand_url
    
, 
    
    
    has_help_center
    
 as 
    
    has_help_center
    
, 
    
    
    help_center_state
    
 as 
    
    help_center_state
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    logo_content_type
    
 as 
    
    logo_content_type
    
, 
    
    
    logo_content_url
    
 as 
    
    logo_content_url
    
, 
    
    
    logo_deleted
    
 as 
    
    logo_deleted
    
, 
    
    
    logo_file_name
    
 as 
    
    logo_file_name
    
, 
    
    
    logo_height
    
 as 
    
    logo_height
    
, 
    
    
    logo_id
    
 as 
    
    logo_id
    
, 
    
    
    logo_inline
    
 as 
    
    logo_inline
    
, 
    
    
    logo_mapped_content_url
    
 as 
    
    logo_mapped_content_url
    
, 
    
    
    logo_size
    
 as 
    
    logo_size
    
, 
    
    
    logo_url
    
 as 
    
    logo_url
    
, 
    
    
    logo_width
    
 as 
    
    logo_width
    
, 
    
    
    name
    
 as 
    
    name
    
, 
    
    
    subdomain
    
 as 
    
    subdomain
    
, 
    
    
    url
    
 as 
    
    url
    




        
, 'google_ads' || '.'|| 'public' as source_relation

        
    from base
),

final as (
    
    select 
        id as brand_id,
        brand_url,
        name,
        subdomain,
        active as is_active,
        source_relation
        
    from fields
    where not coalesce(_fivetran_deleted, false)
)

select * 
from final