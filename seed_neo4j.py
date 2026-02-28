import asyncio
import os
from neo4j import AsyncGraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jpassword")

async def seed_data():
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    query = """
    // 1. Create Client
    MERGE (c:Client {id: 'client_001'})
    SET c.name = 'John Doe'
    
    // 2. Create Risk Profile
    MERGE (rp:RiskProfile {id: 'rp_001'})
    SET rp.profile_type = 'Aggressive',
        rp.risk_tolerance = 'High',
        rp.investment_horizon = 'Long Term (10+ years)'
    MERGE (c)-[:HAS_RISK_PROFILE]->(rp)
    
    // 3. Create Goals
    MERGE (g1:Goal {id: 'goal_001'})
    SET g1.name = 'Retirement', g1.target_amount = 2500000, g1.target_date = '2045-01-01', g1.priority = 'High'
    MERGE (c)-[:HAS_GOAL]->(g1)
    
    MERGE (g2:Goal {id: 'goal_002'})
    SET g2.name = 'Child College Fund', g2.target_amount = 200000, g2.target_date = '2035-09-01', g2.priority = 'Medium'
    MERGE (c)-[:HAS_GOAL]->(g2)
    
    // 4. Create Income Sources
    MERGE (inc:IncomeSource {id: 'inc_001'})
    SET inc.source_type = 'Salary', inc.amount = 185000, inc.frequency = 'Annual'
    MERGE (c)-[:HAS_INCOME_SOURCE]->(inc)
    
    // 5. Create Liabilities
    MERGE (l1:Liability {id: 'liab_001'})
    SET l1.liability_type = 'Mortgage', l1.amount = 450000, l1.interest_rate = 3.5, l1.maturity_date = '2050-06-01'
    MERGE (c)-[:HAS_LIABILITY]->(l1)
    
    // 6. Create Accounts and Assets
    MERGE (acc1:Account {id: 'acc_001'})
    SET acc1.account_type = 'Brokerage', acc1.account_number = 'BRK-998877', acc1.balance = 550000, acc1.currency = 'USD'
    MERGE (c)-[:HAS_ACCOUNT]->(acc1)
    
    MERGE (ass1:Asset {id: 'asset_001'})
    SET ass1.asset_type = 'Stock', ass1.name = 'Apple Inc. (AAPL)', ass1.value = 250000, ass1.quantity = 1500
    MERGE (acc1)-[:CONTAINS]->(ass1)
    
    MERGE (ass2:Asset {id: 'asset_002'})
    SET ass2.asset_type = 'ETF', ass2.name = 'Vanguard S&P 500 (VOO)', ass2.value = 300000, ass2.quantity = 750
    MERGE (acc1)-[:CONTAINS]->(ass2)
    
    RETURN c.id as client_id
    """
    
    async with driver.session() as session:
        result = await session.run(query)
        record = await result.single()
        print(f"Successfully seeded data for client: {record['client_id']}")
        
    await driver.close()

if __name__ == "__main__":
    asyncio.run(seed_data())
