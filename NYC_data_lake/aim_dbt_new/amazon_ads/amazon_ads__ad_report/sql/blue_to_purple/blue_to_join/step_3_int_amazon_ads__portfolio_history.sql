



with portfolios as (
    select
    
        *
        from "amazon_ads"."public_amazon_ads_dev"."stg_amazon_ads__portfolio_history"
        where is_most_recent_record = True
    
)

select * 
from portfolios