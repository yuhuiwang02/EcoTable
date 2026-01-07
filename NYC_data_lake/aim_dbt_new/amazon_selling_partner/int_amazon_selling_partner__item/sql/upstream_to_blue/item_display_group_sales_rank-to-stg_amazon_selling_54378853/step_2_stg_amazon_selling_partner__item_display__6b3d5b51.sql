
-- This model is only necessary when unioning multiple sources and will therefore be disabled when that is not the case






    select
            "asin",
  "website_display_group",
  "title",
  "link",
  "rank",
  "_fivetran_synced"
        from "amazon_selling_partner"."public"."item_display_group_sales_rank" as source_table
    
    