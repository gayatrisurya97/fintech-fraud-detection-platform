-- models/marts/mart_transactions.sql
-- PURPOSE: Final reporting table for Power BI
-- Joins transaction features with customer segments
-- This is what business users and dashboards read from
-- Materialized as a TABLE not a view for better Power BI performance

with features as (

    select * from {{ ref('int_transaction_features') }}

),

customers as (

    select * from {{ ref('int_customer_aggregates') }}

),

final as (

    select
        -- Transaction details
        f.transaction_hour,
        f.transaction_type,
        f.amount,
        f.amount_category,
        f.time_of_day,

        -- Sender details
        f.sender_id,
        f.sender_balance_before,
        f.sender_balance_after,
        f.sender_balance_diff,
        f.sender_balance_zero,

        -- Receiver details
        f.receiver_id,
        f.receiver_balance_before,
        f.receiver_balance_after,
        f.receiver_balance_diff,
        f.receiver_is_merchant,

        -- Fraud indicators
        f.is_fraud,
        f.is_flagged_fraud,
        f.balance_mismatch,

        -- Customer intelligence
        c.total_transactions        as customer_total_transactions,
        c.total_spend               as customer_total_spend,
        c.avg_transaction_amount    as customer_avg_transaction,
        c.fraud_rate_percent        as customer_fraud_rate,
        c.customer_segment,
        c.times_balance_hit_zero    as customer_balance_zero_count

    from features f

    -- Left join so we keep all transactions
    -- even if customer has only 1 transaction
    left join customers c
        on f.sender_id = c.sender_id

)

select * from final