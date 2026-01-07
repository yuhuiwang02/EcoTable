
-- This model is only necessary when unioning multiple sources and will therefore be disabled when that is not the case






    select
            "promotion_id",
  "amazon_order_id",
  "order_item_id"
        from "amazon_selling_partner"."public"."order_item_promotion_id" as source_table
    
    