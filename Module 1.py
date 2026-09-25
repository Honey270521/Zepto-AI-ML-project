import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
for cmd in [[sys.executable,str(root/'data_pipeline'/'pipeline.py')],[sys.executable,str(root/'analytics'/'analysis.py')],[sys.executable,str(root/'support_assistant'/'run_examples.py')]]:
    print('\n>>>', ' '.join(cmd)); subprocess.run(cmd,cwd=root,check=True)
from __future__ import annotations
import sqlite3, re, json
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
BASE_URL='https://books.toscrape.com/'
GBP_TO_INR=105.50
ROOT=Path(__file__).resolve().parent
DB_PATH=ROOT/'books.db'
OUTPUT=ROOT/'cleaned_books.csv'

RATING_MAP={'One':1,'Two':2,'Three':3,'Four':4,'Five':5}

def scrape_books(max_pages=5):
    rows=[]
    try:
        for page in range(1,max_pages+1):
            url=BASE_URL if page==1 else f'{BASE_URL}catalogue/page-{page}.html'
            r=requests.get(url,timeout=15)
            r.raise_for_status()
            soup=BeautifulSoup(r.text,'html.parser')

            for card in soup.select('article.product_pod'):
                title=card.h3.a.get('title','').strip()
                price=card.select_one('.price_color').get_text(strip=True)
                rating_cls=card.select_one('p.star-rating').get('class',[])
                rating_text=next(
                    (x for x in rating_cls if x in RATING_MAP),
                    ''
                )
                availability=card.select_one(
                    '.availability'
                ).get_text(' ',strip=True)
                detail=card.h3.a.get('href')
                detail_url=requests.compat.urljoin(
                    url,
                    detail
                )

                dr=requests.get(detail_url,timeout=15)
                dr.raise_for_status()
                ds=BeautifulSoup(dr.text,'html.parser')
                breadcrumb=[x.get_text(' ',strip=True)
                    for x in ds.select('ul.breadcrumb li')
                ]

                category=(breadcrumb[-1]
                    if len(breadcrumb)>=3
                    else 'Unknown'
                )

                rows.append(
                    dict(
                        title=title,
                        price=price,
                        star_rating=rating_text,
                        availability=availability,
                        category=category
                    )
                )

        if len(rows)<60:
            raise RuntimeError('Live scrape returned fewer than 60 books' )
            return pd.DataFrame(rows), 'live scrape'
            except Exception as exc:
            print(f'Live scrape unavailable;'({exc});'using deterministic offline fixture for local execution.'
        return offline_fixture(), 'offline fixture fallback'

def offline_fixture():
    cats=['Travel','Mystery','Historical Fiction','Classics','Science Fiction','Romance']
    ratings=['One','Two','Three','Four','Five']
    rows=[]

    for i in range(90):
        rows.append(
            {
                'title':f'Practice Book {i+1:03d}',
                'price':f'£{5.00+(i%23)*1.17:.2f}',
                'star_rating':ratings[i%5],
                'availability':'In stock',
                'category':cats[i%len(cats)]
            }
        )

    return pd.DataFrame(rows)

def clean(df):
    out=df.copy()

    out['price_gbp']=pd.to_numeric(
        out['price']
        .astype(str)
        .str.replace(
            '£',
            '',
            regex=False
        )
        .str.strip(),
        errors='coerce'
    )

    out['rating']=out['star_rating'].map(
        RATING_MAP
    )

    out['in_stock']=out['availability'].astype(
        str
    ).str.contains('in stock',
        case=False,
        na=False
    )

    bad_num=(
        out['price_gbp'].isna()
        | out['rating'].isna()
    )

    if out['price_gbp'].isna().any():
        out.loc[
            out['price_gbp'].isna(),
            'price_gbp'
        ] = out['price_gbp'].median()

    if out['rating'].isna().any():
        out.loc[
            out['rating'].isna(),
            'rating'
        ] = round(
            out['rating'].median()
        )

    out=out.dropna(
        subset=['title','category']).copy()

    out['rating']=out['rating'].astype(int)

    out['price_inr']=(
        out['price_gbp']
        * GBP_TO_INR
    ).round(2)

    out['in_stock']=out['in_stock'].astype(bool)

    return (
        out[
            ['title','price_gbp','price_inr','rating','in_stock','category']
        ],
        int(bad_num.sum())
    )


def build_db(df):
    if DB_PATH.exists():
        DB_PATH.unlink()

    con=sqlite3.connect(DB_PATH)

    con.execute('PRAGMA foreign_keys=ON')

    con.executescript(
        '''
        CREATE TABLE categories(category_id INTEGER PRIMARY KEY,category_name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE books(book_id INTEGER PRIMARY KEY,title TEXT NOT NULL,price_gbp REAL NOT NULL,price_inr REAL NOT NULL,rating INTEGER NOT NULL,in_stock INTEGER NOT NULL,category_id INTEGER NOT NULLREFERENCES categories(category_id)
        );
        '''
    )

    cats=sorted(
        df.category.unique()
    )

    con.executemany(
        '''
        INSERT INTO categories(category_name)
        VALUES (?)
        ''',
        [(c,) for c in cats]
    )

    mapping={
        r[1]:r[0]
        for r in con.execute(
            '''
            SELECT category_id,category_name
            FROM categories
            '''
        )
    }

    con.executemany(
        '''
        INSERT INTO books(
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES (?,?,?,?,?,?)
        ''',
        [
            (
                r.title,
                r.price_gbp,
                r.price_inr,
                int(r.rating),
                int(r.in_stock),
                mapping[r.category]
            )
            for r in df.itertuples()
        ]
    )

    con.commit()

    return con


def run_queries(con):
    queries={
        'q1_select_where':
            """
            SELECT title, price_inr
            FROM books
            WHERE price_inr > 1500
            ORDER BY price_inr DESC
            LIMIT 10;
            """,

        'q2_order_limit':
            """
            SELECT title, rating
            FROM books
            ORDER BY rating DESC, title
            LIMIT 10;
            """,

        'q3_distinct':
            """
            SELECT DISTINCT category_name
            FROM categories
            ORDER BY category_name;
            """,

        'q4_between':
            """
            SELECT title, price_gbp
            FROM books
            WHERE price_gbp BETWEEN 10 AND 20
            ORDER BY price_gbp
            LIMIT 10;
            """,

        'q5_join':
            """
            SELECT
                c.category_name,
                b.title,
                b.rating,
                b.price_inr
            FROM books b
            JOIN categories c
                ON b.category_id=c.category_id
            ORDER BY
                b.rating DESC,
                c.category_name,
                b.title
            LIMIT 15;
            """
    }

    results={
        k:
        pd.read_sql_query(
            v,
            con
        ).to_dict(
            orient='records'
        )
        for k,v in queries.items()
    }

    (
        ROOT/'sql_outputs.json'
    ).write_text(
        json.dumps(
            {
                'queries':queries,
                'outputs':results
            },
            indent=2
        )
    )

    return queries,results


def main():
    raw,mode=scrape_books()

    clean_df,bad=clean(raw)

    clean_df.to_csv(
        OUTPUT,
        index=False
    )

    con=build_db(clean_df)

    queries,results=run_queries(con)

    # pd.read_sql and pd.merge equivalence for join
    sql_join=pd.read_sql_query(
        queries['q5_join'],
        con
    )

    books=pd.read_sql_query(
        'SELECT * FROM books',
        con
    )

    cats=pd.read_sql_query(
        'SELECT * FROM categories',
        con
    )

    merged=(
        books
        .merge(
            cats,
            on='category_id'
        )
        .rename(
            columns={
                'category_name':'category'
            }
        )
        [
            [
                'category',
                'title',
                'rating',
                'price_inr'
            ]
        ]
        .sort_values(
            [
                'rating',
                'category',
                'title'
            ],
            ascending=[
                False,
                True,
                True
            ]
        )
        .head(15)
        .reset_index(drop=True)
    )

    sql_cmp=sql_join.reset_index(
        drop=True
    )

    sql_cmp.columns=[
        'category',
        'title',
        'rating',
        'price_inr'
    ]

    eq=sql_cmp.round(6).equals(
        merged.round(6)
    )

    (
        ROOT/'run_summary.md'
    ).write_text(
        f'''
# Data Pipeline Run Summary

- Mode: {mode}
- Rows: {len(clean_df)}
- Categories: {clean_df.category.nunique()}
- Parse issues handled: {bad}
- Fixed rate: 1 GBP = {GBP_TO_INR:.2f} INR
- SQLite: `{DB_PATH.name}`
- SQL/Pandas JOIN equivalence: **{eq}**
'''
    )

    print(
        f'Data pipeline complete: '
        f'{len(clean_df)} rows, '
        f'{clean_df.category.nunique()} categories, '
        f'JOIN equivalent={eq}'
    )
  con.close()

if __name__=='__main__':
    main()