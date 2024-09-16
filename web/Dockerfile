FROM node:latest as build

WORKDIR /app

COPY package*.json ./

RUN npm install

RUN npx playwright install
RUN npx playwright install-deps
RUN npm install uuid
RUN npm install js-yaml
RUN npm install playwright-extra puppeteer-extra-plugin-stealth


COPY . .

CMD ["npm", "start"]
