with base as (

    select * 
    from "xero"."public_xero_dev"."stg_xero__contact_tmp"

),

fields as (

    select
        
    
    
    _fivetran_deleted
    
 as 
    
    _fivetran_deleted
    
, 
    cast(null as timestamp) as 
    
    _fivetran_synced
    
 , 
    cast(null as TEXT) as 
    
    account_number
    
 , 
    cast(null as TEXT) as 
    
    accounts_payable_tax_type
    
 , 
    cast(null as TEXT) as 
    
    accounts_receivable_tax_type
    
 , 
    cast(null as numeric(28,6)) as 
    
    balances_accounts_payable_outstanding
    
 , 
    cast(null as numeric(28,6)) as 
    
    balances_accounts_payable_overdue
    
 , 
    cast(null as numeric(28,6)) as 
    
    balances_accounts_receivable_outstanding
    
 , 
    cast(null as numeric(28,6)) as 
    
    balances_accounts_receivable_overdue
    
 , 
    cast(null as TEXT) as 
    
    bank_account_details
    
 , 
    cast(null as TEXT) as 
    
    batch_payments_bank_account_name
    
 , 
    cast(null as TEXT) as 
    
    batch_payments_bank_account_number
    
 , 
    cast(null as TEXT) as 
    
    batch_payments_code
    
 , 
    cast(null as TEXT) as 
    
    batch_payments_details
    
 , 
    cast(null as TEXT) as 
    
    batch_payments_reference
    
 , 
    cast(null as TEXT) as 
    
    branding_theme_id
    
 , 
    
    
    contact_id
    
 as 
    
    contact_id
    
, 
    cast(null as TEXT) as 
    
    contact_number
    
 , 
    cast(null as TEXT) as 
    
    contact_status
    
 , 
    cast(null as TEXT) as 
    
    default_currency
    
 , 
    cast(null as integer) as 
    
    discount
    
 , 
    cast(null as TEXT) as 
    
    email_address
    
 , 
    cast(null as TEXT) as 
    
    first_name
    
 , 
    cast(null as boolean) as 
    
    has_attachments
    
 , 
    cast(null as boolean) as 
    
    has_validation_errors
    
 , 
    cast(null as boolean) as 
    
    is_customer
    
 , 
    cast(null as boolean) as 
    
    is_supplier
    
 , 
    cast(null as TEXT) as 
    
    last_name
    
 , 
    
    
    name
    
 as 
    
    name
    
, 
    cast(null as TEXT) as 
    
    purchases_default_account_code
    
 , 
    cast(null as TEXT) as 
    
    sales_default_account_code
    
 , 
    cast(null as TEXT) as 
    
    skype_user_name
    
 , 
    cast(null as TEXT) as 
    
    tax_number
    
 , 
    cast(null as timestamp) as 
    
    updated_date_utc
    
 , 
    cast(null as TEXT) as 
    
    website
    
 , 
    cast(null as TEXT) as 
    
    xero_network_key
    
 



        



        
    from base
),

final as (
    
    select 
        contact_id,
        name as contact_name

        


, cast('' as TEXT) as source_relation



        
    from fields
    where _fivetran_deleted = False
)

select * from final