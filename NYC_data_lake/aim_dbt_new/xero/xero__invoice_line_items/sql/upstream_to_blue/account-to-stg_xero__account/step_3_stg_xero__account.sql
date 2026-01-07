with base as (

    select * 
    from "xero"."public_xero_dev"."stg_xero__account_tmp"

),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    account_id
    
 as 
    
    account_id
    
, 
    cast(null as TEXT) as 
    
    bank_account_number
    
 , 
    cast(null as TEXT) as 
    
    bank_account_type
    
 , 
    
    
    class
    
 as 
    
    class
    
, 
    
    
    code
    
 as 
    
    code
    
, 
    cast(null as TEXT) as 
    
    currency_code
    
 , 
    cast(null as TEXT) as 
    
    description
    
 , 
    cast(null as boolean) as 
    
    enable_payments_to_account
    
 , 
    cast(null as boolean) as 
    
    has_attachments
    
 , 
    
    
    name
    
 as 
    
    name
    
, 
    cast(null as TEXT) as 
    
    reporting_code
    
 , 
    cast(null as TEXT) as 
    
    reporting_code_name
    
 , 
    cast(null as boolean) as 
    
    show_in_expense_claims
    
 , 
    cast(null as TEXT) as 
    
    status
    
 , 
    cast(null as TEXT) as 
    
    system_account
    
 , 
    cast(null as TEXT) as 
    
    tax_type
    
 , 
    
    
    type
    
 as 
    
    type
    
, 
    cast(null as timestamp) as 
    
    updated_date_utc
    
 



        




    from base
),

final as (
    
    select 
        account_id,
        name as account_name,
        code as account_code,
        type as account_type,
        class as account_class,
        _fivetran_synced

        


, cast('' as TEXT) as source_relation




    from fields

)

select * from final