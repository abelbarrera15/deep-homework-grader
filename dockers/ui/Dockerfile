# pull official base image
FROM node:13.12.0-alpine

#install build serviture utility
RUN npm install -g serve --silent

# set working directory
WORKDIR /app

# add `/app/node_modules/.bin` to $PATH
ENV PATH /app/node_modules/.bin:$PATH

# install app dependencies
COPY package.json package-lock.json ./
RUN npm install --silent

# copy app files into Docker
COPY . /app

# build app
RUN npm run build

# set app serving port
EXPOSE 3000

#serve app
CMD ["serve", "build"]