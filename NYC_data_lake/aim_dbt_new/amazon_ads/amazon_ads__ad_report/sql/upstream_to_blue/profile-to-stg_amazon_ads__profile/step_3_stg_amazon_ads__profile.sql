

with base as (

    select * 
    from "amazon_ads"."public_amazon_ads_dev"."stg_amazon_ads__profile_tmp"
),

fields as (

    select
        
    
    
    id
    
 as 
    
    id
    
, 
    
    
    account_id
    
 as 
    
    account_id
    
, 
    
    
    account_marketplace_string_id
    
 as 
    
    account_marketplace_string_id
    
, 
    
    
    account_name
    
 as 
    
    account_name
    
, 
    
    
    account_sub_type
    
 as 
    
    account_sub_type
    
, 
    
    
    account_type
    
 as 
    
    account_type
    
, 
    
    
    account_valid_payment_method
    
 as 
    
    account_valid_payment_method
    
, 
    
    
    country_code
    
 as 
    
    country_code
    
, 
    
    
    currency_code
    
 as 
    
    currency_code
    
, 
    
    
    daily_budget
    
 as 
    
    daily_budget
    
, 
    
    
    timezone
    
 as 
    
    timezone
    
, 
    
    
    _fivetran_deleted
    
 as 
    
    _fivetran_deleted
    



    
        


, cast('' as TEXT) as source_relation




    from base
),

final as (

    select
        source_relation, 
        cast(id as TEXT) as profile_id,
        cast(account_id as TEXT) as account_id,
        account_marketplace_string_id,
        account_name,
        account_sub_type,
        account_type,
        account_valid_payment_method,
        country_code,
        currency_code,
        daily_budget,
        timezone,
        _fivetran_deleted
    from fields
)

select *
from final