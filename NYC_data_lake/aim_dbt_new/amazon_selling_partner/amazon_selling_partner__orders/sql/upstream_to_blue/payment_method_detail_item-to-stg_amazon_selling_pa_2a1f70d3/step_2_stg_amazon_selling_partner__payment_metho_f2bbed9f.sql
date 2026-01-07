
-- This model is only necessary when unioning multiple sources and will therefore be disabled when that is not the case






    select
            "amazon_order_id",
  "method"
        from "amazon_selling_partner"."public"."payment_method_detail_item" as source_table
    
    