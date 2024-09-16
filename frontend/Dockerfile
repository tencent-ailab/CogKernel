
FROM node:latest as build


WORKDIR /app


COPY package*.json ./


RUN npm install
RUN npm install antd-img-crop --save
RUN npm install react-json-view --legacy-peer-deps


COPY . .

RUN npm run build

FROM nginx:alpine

RUN apk add --no-cache bash

COPY --from=build /app/dist /usr/share/nginx/html

COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

